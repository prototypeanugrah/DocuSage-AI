import argparse

import PyPDF2


def contains_images_pdf(pdf_path):
    """Checks if a PDF contains images using PyPDF2."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                # Check for XObject resources, which often contain images
                if "/XObject" in page["/Resources"]:
                    xobjects = page["/Resources"]["/XObject"].get_object()
                    for obj in xobjects:
                        if xobjects[obj]["/Subtype"] == "/Image":
                            return True
        return False
    except Exception as e:
        print(f"Error processing PDF with PyPDF2: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if a PDF contains images.")
    parser.add_argument("--input", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    if contains_images_pdf(args.input):
        print(f"The PDF '{args.input}' contains images.")
    else:
        print(f"The PDF '{args.input}' does not contain images.")
