import base64
import io
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def parse_document(file_path: str) -> list[dict]:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    result = converter.convert(file_path)
    doc = result.document

    parsed_chunks: list[dict] = []
    current_section: str | None = None
    source_file = os.path.basename(file_path)

    for item in doc.iterate_items():
        if isinstance(item, tuple):
            node, _ = item
        else:
            node = item

        label = str(getattr(node, "label", "")).lower()

        if label in ("page_header", "page_footer"):
            continue

        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None
        position: dict | None = None
        if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        def _make_metadata(content_type: str, element_type: str, img_b64=None):
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,
                "image_base64": img_b64,
            }

        if "section_header" in label or label == "title":
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": _make_metadata("text", label),
                    }
                )

        elif "table" in label:
            table_text = ""
            if hasattr(node, "export_to_dataframe"):
                try:
                    df = node.export_to_dataframe()
                    if df is not None and not df.empty:
                        rows_text: list[str] = []
                        headers = [str(c).strip() for c in df.columns]
                        for _, row in df.iterrows():
                            pairs = [
                                f"{h}: {str(v).strip()}"
                                for h, v in zip(headers, row)
                                if str(v).strip() not in ("", "nan", "None")
                            ]
                            if pairs:
                                rows_text.append("  |  ".join(pairs))
                        table_text = "\n".join(rows_text)
                except Exception:
                    pass

            if not table_text and hasattr(node, "export_to_html"):
                try:
                    import re as _re
                    raw_html = node.export_to_html(doc)
                    table_text = _re.sub(r"<[^>]+>", " ", raw_html or "")
                    table_text = _re.sub(r"\s+", " ", table_text).strip()
                except Exception:
                    pass

            if not table_text:
                table_text = getattr(node, "text", "")

            if table_text and table_text.strip():
                parsed_chunks.append(
                    {
                        "content": table_text.strip(),
                        "content_type": "table",
                        "metadata": _make_metadata("table", "table"),
                    }
                )

        elif "picture" in label or "figure" in label or label == "chart":
            img_b64 = None
            caption = getattr(node, "text", "") or ""

            try:
                if hasattr(node, "get_image"):
                    pil_img = node.get_image(doc)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()

                if img_b64 is None and hasattr(node, "image") and node.image:
                    pil_img = getattr(node.image, "pil_image", None)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass

            content = caption.strip() or f"[Image on page {page_no}]"
            parsed_chunks.append(
                {
                    "content": content,
                    "content_type": "image",
                    "metadata": _make_metadata("image", "picture", img_b64),
                }
            )

        else:
            text = getattr(node, "text", "")
            if text and text.strip():
                parsed_chunks.append(
                    {
                        "content": text.strip(),
                        "content_type": "text",
                        "metadata": _make_metadata("text", label),
                    }
                )

    return parsed_chunks