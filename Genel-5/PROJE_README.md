# Advanced Image Processing Application

This project is a Streamlit application that splits large images into smaller patches, applies a variety of filters to those patches, and then stitches the image back together.

## Features

- Split large images into tiles
- Apply different image filters to each tile
- Merge processed tiles back to the original resolution
- User-friendly interface
- Option to download the processed image

## Installation

1. Install the required libraries:
   ```
   pip install streamlit numpy torch Pillow
   ```

2. Run the application:
   ```
   streamlit run image_processor_app.py
   ```

## Usage

1. Upload an image from the menu on the left
2. Choose the filter you want to apply
3. Adjust the overlap ratio and maximum number of patches
4. Click the "Process Image" button
5. Review and download the processed image

## Available Filters

- Normal: Original image
- Black & White: Grayscale conversion
- Blur: Blurring effect
- Contour: Edge detection
- Sharpen: Enhance image sharpness

## Development

This project demonstrates the workflow for splitting and recombining large images. You can extend it with additional features, such as:

- Additional filter options
- Customizable patch sizes
- Batch processing support
- Alternative export formats

## License

MIT
