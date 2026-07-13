import cv2
import argparse
import os
import subprocess

#Image Aquisition
def capture_image(output_path="my_image.jpg"):
    print(f"Capturing image with raspistill -> {output_path}")
    subprocess.run(["raspistill", "-o", output_path], check=True)
    print("Capture done.")
    return output_path
def load_image(image_path):
     print(f"Loading image from {image_path}")
     image = cv2.imread(image_path)
     if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
     return image
#Grayscale Conversion
def convert_to_grayscale(image):
    print("Converting image to grayscale.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

#Gaussian Blur
def apply_gaussian_blur(gray_image):
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    return blurred_image

#canny Edge Detection
def apply_canny_edge_detection(blurred_image):
    edges = cv2.Canny(blurred_image, 10, 250)
    return edges

#morphological Operations
def apply_morphological_operations(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return morphed_image

#contour Detection
def detect_contours(morphed_image):
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

#shape classification
def classify_shapes(contour):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    if vertices == 3:
        shape = "triangular"
    elif vertices == 4:
        shape = "rectangular"
    else:
        shape = "polygon"   #circle or other shapes can be classified as polygon for simplicity
    return shape, approx

#draw contours and classify shapes
def draw_contours_and_classify(image, contours, output_path=None):
    print("Drawing contours and classifying shapes.")
    output = image.copy()
    triangles  = 0
    rectangles = 0
    polygons   = 0

    for contour in contours:
        shape, approx = classify_shapes(contour)
        if shape == "triangular":
            color = (0, 255, 0)      # green
            triangles += 1
        elif shape == "rectangular":
            color = (0, 0, 255)      # red
            rectangles += 1
        else:
            color = (255, 0, 0)      # blue
            polygons += 1
        cv2.drawContours(output, [approx], -1, color, 2)

    total = triangles + rectangles + polygons

    print("\n" + "=" * 52)
    print("  DETECTION RESULTS")
    print("=" * 52)

    # triangles
    if triangles == 0:
        print("There isn't any triangular object in the image.")
    elif triangles == 1:
        print("There is 1 triangular object in the image.")
    else:
        print(f"There are {triangles} triangular objects in the image.")

    # rectangles
    if rectangles == 0:
        print("There isn't any rectangular object in the image.")
    elif rectangles == 1:
        print("There is 1 rectangular object in the image.")
    else:
        print(f"There are {rectangles} rectangular objects in the image.")

    # polygons
    if polygons == 0:
        print("There isn't any object with more than four edges in the image.")
    elif polygons == 1:
        print("There is 1 object with more than four edges in the image.")
    else:
        print(f"There are {polygons} objects with more than four edges in the image.")

    # total — paper uses this exact cheerful phrasing
    if total == 1:
        print(f"\nI am glad to let you know that there is 1 object in this image.")
    else:
        print(f"\nI am glad to let you know that there are {total} objects in this image.")

    print("=" * 52)
    # save output image
    if output_path:
        cv2.imwrite(output_path, output)
        print(f"\nOutput saved to: {output_path}")
    return output, triangles, rectangles, polygons



def main():

    parser = argparse.ArgumentParser(
        description="Baseline re-implementation of Abdulhamid et al. (2020)"
    )
    parser.add_argument(
        "--image",  
        type=str,
        default="my_image.jpg",
        help="Path to input image (e.g. my_image.jpg)"
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Show each pipeline step visually"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output image to file"
    )
    args = parser.parse_args()

    print("\n" + "=" * 52)
    print("  Baseline Re-implementation")
    print("  Abdulhamid, Odondi & Al-Rawi (2020)")
    print("=" * 52)

    # determine output path
    output_path = None
    if args.save:
        name = os.path.splitext(args.image)[0]
        output_path = f"{name}_output.jpg"

    # ── run pipeline ──────

    # Step 1 — image acquisition
    
    image_path = args.image
    if args.show_steps:
     print("\nStep 1: Loading image...")
     image = load_image(image_path)

    # Step 2 — grayscale
     print("Step 2: Converting to grayscale...")
     gray = convert_to_grayscale(image)

    # Step 3 — gaussian blur
     print("Step 3: Applying Gaussian blur (3x3, sigma=0)...")
     blurred = apply_gaussian_blur(gray)

    # Step 4 — canny edge detection
     print("Step 4: Detecting edges (Canny, min=10, max=250)...")
     edged = apply_canny_edge_detection(blurred)

    # Step 5 — morphological closing
     print("Step 5: Applying morphological closing (7x7 kernel)...")
     closed = apply_morphological_operations(edged)

    # Step 6 — find contours
     print("Step 6: Finding contours...")
     contours = detect_contours(closed)
     print(f"        {len(contours)} raw contours found")

    # Step 7+8 — classify and display
     print("Step 7: Classifying shapes and counting objects...")
     output, tri, rect, poly = draw_contours_and_classify(
        image, contours, output_path
     )
    else:
     image = load_image(image_path)
     gray = convert_to_grayscale(image)
     blurred = apply_gaussian_blur(gray)
     edged = apply_canny_edge_detection(blurred)
     closed = apply_morphological_operations(edged)
     contours = detect_contours(closed)
     output, tri, rect, poly = draw_contours_and_classify(image, contours, output_path) 
    max_w, max_h = 800, 600
    h, w = output.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    display = cv2.resize(output, (int(w * scale), int(h * scale)))

    cv2.imshow("Output Baseline Detection", display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    