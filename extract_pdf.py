import PyPDF2

# Read the PDF and extract text
with open("Planning_Report_Group13.pdf", "rb") as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Save to a text file for easier reading
    with open("planning_report.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(text)

    print("PDF content extracted successfully!")
    print(f"Total pages: {len(reader.pages)}")
