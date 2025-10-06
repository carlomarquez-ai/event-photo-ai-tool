# Event Photo Auto-Processor

A Python-based tool to automatically cull and enhance event photos.  
Removes blurry, low-quality, and duplicate images while applying basic edits to produce a polished final batch. Ideal for photographers who want to save time and maintain consistent quality.

## Features

- **Auto-Culling**: Detects and removes blurry, low-quality, and duplicate photos
- **Face & Smile Prioritization**: Optionally prioritize images with clear faces or smiles
- **Basic Enhancements**: Adjusts exposure, white balance, contrast, and cropping
- **Stylized Presets**: Apply consistent style across all photos
- **Optional Watermark**: Add logos or text overlays
- **Local Execution**: Runs on Windows or macOS via CLI or a simple GUI
- **Logging**: Generates a summary log of processed images

## Tech Stack

- Python
- OpenCV – image processing
- Pillow – image manipulation
- Hugging Face models (optional) – image quality or aesthetic scoring

## Usage

1. Place raw event photos in a folder
2. Run the tool (CLI or GUI)
3. Processed images are saved in a `Processed` folder with a log summary

## Benefits

- Saves hours of manual photo selection and editing
- Ensures consistent image quality and style across events
- Customizable for client-specific preferences

## License
MIT License