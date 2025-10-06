from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("GENERAL_PROVISION.pdf").document

markdown_output = doc.export_to_markdown()
print(markdown_output)