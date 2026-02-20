import time
import pynvml

def get_gpu_energy(handle):
    """Get total energy consumption (milli-joules) for the GPU."""
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

def main():
    # Initialize NVML
    pynvml.nvmlInit()

    # Select first GPU (index 0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"Measuring energy on GPU 0: {name.decode() if isinstance(name, bytes) else name}")

    # Ask how many seconds to measure
    seconds = float(input("Enter duration in seconds to measure base energy: "))

    start_energy = get_gpu_energy(handle)
    print(f"Starting energy: {start_energy} mJ")

    time.sleep(seconds)

    end_energy = get_gpu_energy(handle)
    print(f"Ending energy: {end_energy} mJ")

    energy_used = end_energy - start_energy
    print(f"\nEnergy consumed over {seconds} seconds: {energy_used} mJ "
          f"({energy_used/1000:.3f} J)")

    # Shutdown NVML
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
