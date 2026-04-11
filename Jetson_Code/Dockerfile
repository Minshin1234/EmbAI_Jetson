# Use the official NVIDIA L4T-ML image as the base
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

WORKDIR /app

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/aarch64-linux-gnu/tegra"

# Install system tools and GStreamer plugins
RUN apt-get update && apt-get install -y --no-install-recommends \
    kmod \
    i2c-tools \
    libegl1 \
    libgles2 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# Fix the blinker package that blocks pip (force-replace it alone)
RUN pip3 install --no-cache-dir --ignore-installed blinker

# Install Flask, IK solver, serial bridge, and safe utilities
RUN pip3 install --no-cache-dir flask psutil py-cpuinfo tqdm pillow ikpy pyserial

# Install ultralytics WITHOUT dependencies (torch/torchvision/opencv/numpy stay native)
RUN pip3 install --no-cache-dir --no-deps ultralytics ultralytics-thop

CMD ["python3", "stream.py"]
