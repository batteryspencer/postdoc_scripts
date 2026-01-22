import os, base64
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]  # or set directly
client = Mistral(api_key=api_key)

pdf_path = "input.pdf"

# encode local PDF to a data: URL the API accepts
with open(pdf_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{b64}",
    },
    # optional:
    # pages=[0, 1, 2],  # 0-indexed pages
    # include_image_base64=True,
)

print(ocr_response)
