###
# vllm_serving.py
#
# Routines for MoE experiments using the vLLM serving interface:
# server lifecycle, profiling control, request timing and expert capture.
# Dylan Everingham
# 06.08.2026
###

import time, subprocess, os, signal, asyncio
import io, base64
import urllib.request
import urllib.error
import numpy as np
from openai import AsyncOpenAI

# Spins up the vLLM server as a subprocess and blocks until ready.
def start_vllm_server(model_name, port=8000, seed=0, max_model_len=1024, batch_size=16, gpu_memory_utilization=0.85, n_gpus=1, enable_bnb=False, enable_expert_parallel=False, enable_prefix_caching=False, enable_eplb=False, enable_expert_capture=False, trace_dir=None, trace_start_iteration=50, trace_active_iterations=10, startup_timeout=1800):
    print(f"Starting vLLM server for {model_name}...")
    
    cmd = [
        "vllm", "serve", model_name,
        "--port", str(port),
        "--dtype", "auto",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(batch_size), # Batch size is used to set max number of batched requests
        "--max-num-batched-tokens", "4096", # Max number of tokens per forward pass is fixed based on hardware
        "--tensor-parallel-size", str(n_gpus),
        "--data-parallel-size", "1",
        "--seed", str(seed),
        "--override-generation-config", '{"temperature": 0.0}',
        "--enforce-eager",
        "--no-async-scheduling",
    ]
    
    if trace_dir:
        cmd.append("--profiler-config")
        cmd.append(f'{{"profiler": "torch", "torch_profiler_dir": "{trace_dir}", "active_iterations": {trace_active_iterations}, "delay_iterations": {trace_start_iteration}, "torch_profiler_with_stack": false}}')
    
    if enable_expert_parallel:
        cmd.append("--enable-expert-parallel")

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")

    if enable_bnb:
         cmd.extend(["--quantization", "bitsandbytes"])

    if enable_eplb:
        cmd.append("--enable-eplb")

    if enable_expert_capture:
        cmd.append("--enable-return-routed-experts")

        
    # Own session, so the whole server tree (API server, engine core, workers)
    # can be signalled as one group on teardown.
    server_process = subprocess.Popen(cmd, start_new_session=True)
    
    # Poll the endpoint for 200 OK. Bounded: a server that comes up but never
    # answers leaves this loop spinning forever while holding every NPU of the
    # run, which is far worse than failing the point.
    print("Waiting for server to initialize ...")
    url = f"http://localhost:{port}/v1/models"
    deadline = time.monotonic() + startup_timeout

    while True:
        try:
            response = urllib.request.urlopen(url, timeout=10)
            if response.getcode() == 200:
                print("Server is ready!")
                break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)

        if server_process.poll() is not None:
            raise RuntimeError("vLLM server process terminated unexpectedly.")

        if time.monotonic() > deadline:
            stop_vllm_server(server_process)
            raise RuntimeError(f"vLLM server did not become ready within {startup_timeout:.0f}s "
                               f"(it was still running; see the log for where it stopped)")

    return server_process

# Terminates the vLLM server subprocess tree and waits for the NPUs to be released.
# Signalling only the API server leaves the engine core and worker processes alive
# briefly; they hold the HBM, so the next sweep point can fail to allocate.
def stop_vllm_server(server_process, timeout=120.0):
    print("Shutting down vLLM server...")

    try:
        pgid = os.getpgid(server_process.pid)
    except ProcessLookupError:
        server_process.wait()
        print("Server successfully shut down.")
        return

    def group_alive():
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        return True

    def wait_for_group(deadline):
        while time.monotonic() < deadline:
            if not group_alive():
                return True
            time.sleep(1)
        return not group_alive()

    os.killpg(pgid, signal.SIGTERM)
    server_process.wait()  # reap the parent so it stops counting as a group member

    if not wait_for_group(time.monotonic() + timeout):
        print(f"Server tree still alive after {timeout:.0f}s, sending SIGKILL...")
        os.killpg(pgid, signal.SIGKILL)
        if not wait_for_group(time.monotonic() + 30.0):
            raise RuntimeError("vLLM server processes did not exit; NPU memory may still be held.")

    print("Server successfully shut down.")

def start_profiling(port=8000, timeout=600):
    # Needs a timeout like stop_profiling: urlopen without one waits forever, and
    # a profiler that never answers then holds every NPU of the run indefinitely.
    print("Starting vLLM PyTorch Profiler...")
    req = urllib.request.Request(f"http://localhost:{port}/start_profile", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            print("Profiler started successfully.")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"Failed to start profiler: {e}")

def stop_profiling(port=8000, timeout=600):
    print("Stopping vLLM PyTorch Profiler (Note: flushing traces to disk may take a few minutes)...")
    req = urllib.request.Request(f"http://localhost:{port}/stop_profile", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            print("Profiler stopped and traces flushed successfully.")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"Failed to stop profiler: {e}")


# Helper to parse response containing routed expert info
def decode_routed_experts(payload):
    if payload is None:
        return None
    if isinstance(payload, np.ndarray):
        return payload.astype(np.int16, copy=False)
    if isinstance(payload, str):
        buf = io.BytesIO(base64.b64decode(payload))
        return np.load(buf, allow_pickle=False).astype(np.int16, copy=False)
    return np.asarray(payload, dtype=np.int16)

# Sends a single streaming request and measures TTFT and TPOT
async def measure_request(client, model, prompt_idx, prompt, seed=0, max_new_tokens=100,
                          get_response=False, prompt_formatted=True, capture_experts=False):
    
    start_time = time.perf_counter()
    first_token_time = None
    if prompt_formatted:
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    # If recording expert activations...
    if capture_experts:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_new_tokens,
            seed=seed,
            temperature=0
        )
        end_time = time.perf_counter()
        choice = response.choices[0]
        routed_experts = decode_routed_experts(getattr(choice, "routed_experts", None))
        prompt_routed_experts = decode_routed_experts(getattr(response, "prompt_routed_experts", None))
        num_input_tokens = response.usage.completion_tokens
        num_output_tokens = response.usage.prompt_tokens

        if prompt_routed_experts is None and routed_experts is not None and routed_experts.shape[0] > num_input_tokens:
            prompt_routed_experts = routed_experts[:num_input_tokens]
            routed_experts = routed_experts[num_input_tokens:]

        result = {
            "prompt": prompt,
            "prompt_id": prompt_idx,
            "ttft": None, # Don't report times when recording expert activations, as they are not valid
            "tpot": None,
            "num_output_tokens": num_input_tokens,
            "num_input_tokens": num_output_tokens,
            "total_time": end_time - start_time,
            # [gen_len, n_moe_layers, top_k] and [prompt_len, n_moe_layers, top_k]
            "routed_experts": routed_experts,
            "prompt_routed_experts": prompt_routed_experts,
        }
        if get_response:
            result["response"] = choice.message.content
        return result

    # Otherwise, not capturing experts
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_new_tokens,
        seed=seed,
        temperature=0
    )

    response_str = ""
    num_output_tokens = 0
    
    async for chunk in response:
        # Get output tokens
        if get_response:
            if chunk.choices and chunk.choices[0].delta.content:
                response_str += chunk.choices[0].delta.content
        
        # Time of first token (in order to deduct decode fromm TPOT)
        if first_token_time is None and chunk.choices:
            first_token_time = time.perf_counter()
            
        # The last chunk when using include_usage=True contains the token stats
        if chunk.usage is not None:
            num_output_tokens = chunk.usage.completion_tokens
            num_input_tokens = chunk.usage.prompt_tokens

    end_time = time.perf_counter()

    # If generation failed or no tokens produced...
    if first_token_time is None:
        first_token_time = end_time

    # Calculate metrics
    ttft = first_token_time - start_time
    generation_time = end_time - first_token_time
    
    # Subtract 1 from output_tokens because the first token's time is captured in TTFT
    tpot = 0
    if num_output_tokens > 1:
        tpot = generation_time / (num_output_tokens - 1)
    
    result = {
        "prompt": prompt,
        "prompt_id": prompt_idx,
        "ttft": ttft,
        "tpot": tpot,
        "num_output_tokens": num_output_tokens,
        "num_input_tokens": num_input_tokens,
        "total_time": end_time - start_time
    }
    if get_response:
        result["response"] = response_str
    return result
    
# Runs a batch of prompts concurrently and calculates aggregate metrics
async def run_batch(client, model, prompts, seed=0, max_new_tokens=100, concurrency_limit=100, print_output=False, prompt_formatted=True, capture_experts=False):

    print(f"Sending batch of {len(prompts)} concurrent requests...")
    
    batch_start_time = time.perf_counter()

    # Use semaphore to limit request rate
    # pipeline should still saturate as long as concurrency_limit > max_num_seqs (batch size)
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Wrapper function that acquires the semaphore before making the request
    async def rate_limited_measure_request(i, prompt):
        async with semaphore:
            return await measure_request(
                client, 
                model, 
                i, 
                prompt, 
                seed=seed, 
                max_new_tokens=max_new_tokens, 
                get_response=False,
                prompt_formatted=prompt_formatted,
                capture_experts=capture_experts
            )
    
    # Fire all requests
    tasks = [rate_limited_measure_request(i, prompt) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    batch_end_time = time.perf_counter()
    total_batch_time = batch_end_time - batch_start_time
    
    if print_output:
        print("\n--- Per-Request Metrics ---")
    total_tpot = 0
    total_tokens = 0
    valid_requests = 0

    for res in results:
        if print_output and res['ttft'] is not None:
            print(f"Request {res['prompt_id']}: TTFT = {res['ttft']:.4f}s | "
                  f"TPOT = {res['tpot']*1000:.2f}ms | Tokens = {res['num_output_tokens']}")

        if res['num_output_tokens'] > 1 and res['tpot'] is not None:
            total_tpot += res['tpot']
            total_tokens += res['num_output_tokens']
            valid_requests += 1
            
        if print_output:
            print("\n--- Batch Metrics ---")
        if valid_requests > 0:
            avg_tpot = total_tpot / valid_requests
            if print_output:
                print(f"Average Per-Request TPOT: {avg_tpot * 1000:.2f} ms/token")

        throughput = total_tokens / total_batch_time
        if print_output:
            print(f"Total Batch Time: {total_batch_time:.2f}s")
            print(f"Total Tokens Generated: {total_tokens}")
            print(f"Overall Server Throughput: {throughput:.2f} tokens/second")
    
    return results 

# Run full experiment:
# - Start vLLM server
# - Start vLLM client
# - Run inference
# - Return timing measurements
async def measure_vllm_throughput(model, prompts, seed=0, max_new_tokens=100, concurrency_limit=1024,
                                  max_model_len=1024, batch_size=256, gpu_memory_utilization=0.85,
                                  n_gpus=1, n_warmup_samples=5,
                                  print_output=False, enable_bnb=False, enable_expert_parallel=False,
                                  enable_prefix_caching=False, enable_eplb=False, enable_expert_capture=False,
                                  trace_dir=None, trace_active_iterations=2, port=8000):
    server_process = None
    results = None
    try:
        # Start server
        server_process = start_vllm_server(model, port=port, seed=seed,
                                           max_model_len=max_model_len,
                                           batch_size=batch_size,
                                           gpu_memory_utilization=gpu_memory_utilization,
                                           n_gpus=n_gpus, enable_expert_parallel=enable_expert_parallel,
                                           enable_prefix_caching=enable_prefix_caching, enable_bnb=enable_bnb,
                                           enable_eplb=enable_eplb,
                                           enable_expert_capture=enable_expert_capture,
                                           trace_dir=trace_dir, trace_start_iteration=100,
                                           trace_active_iterations=trace_active_iterations)

        # Start client
        client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    
        # Run warmup
        await run_batch(client, model, prompts[:n_warmup_samples],
                        seed=seed, print_output=False, max_new_tokens=max_new_tokens,
                        concurrency_limit=concurrency_limit,
                        prompt_formatted=True)
        
        # Start profiling
        if trace_dir is not None:
            start_profiling(port=port)
        
        # Run experiment
        results = await run_batch(client, model, prompts,
                                  seed=seed, print_output=print_output, max_new_tokens=max_new_tokens,
                                  concurrency_limit=concurrency_limit, capture_experts=enable_expert_capture,
                                  prompt_formatted=True)
        
        # Stop profiling
        if trace_dir is not None:
            stop_profiling(port=port)

    except Exception as e:
        print(f"An error occurred during inference: {e}")

    finally:
        # Tear down server
        if server_process is not None:
            stop_vllm_server(server_process)

    return results
