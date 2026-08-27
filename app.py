from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_annotation import detection

from flyer_pipeline import MODEL_ID, process_flyer, render_pdf, valid_bbox


st.set_page_config(page_title="Flyer AI Reader", layout="wide")
st.title("Flyer AI Reader")
st.caption(
    "Local review dashboard for the GA capstone. "
    "This avoids Streamlit Cloud/OpenRouter authentication issues."
)

DB_PATH = Path("flyer_data.db")
CORRECTIONS_PATH = Path("corrections.csv")


def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            reviewed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            shop_name TEXT,
            campaign_name TEXT,
            product_count INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approved_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            shop_name TEXT,
            campaign_name TEXT,
            region TEXT,
            branch TEXT,
            flyer_start_date TEXT,
            flyer_end_date TEXT,
            page INTEGER,
            product_name TEXT,
            quantity TEXT,
            price_before REAL,
            price_after REAL,
            currency TEXT,
            product_start_date TEXT,
            product_end_date TEXT,
            date_source TEXT,
            date_badge_text TEXT,
            bbox TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def draw_bbox(image_path, bbox):
    image = Image.open(image_path).convert("RGB")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return image

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return image

    width, height = image.size
    pixel_box = (
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    )

    draw = ImageDraw.Draw(image)
    line_width = max(3, int(min(width, height) * 0.006))
    draw.rectangle(pixel_box, outline="red", width=line_width)
    return image


def editable_columns():
    return [
        "product_name",
        "quantity",
        "price_before",
        "price_after",
        "currency",
        "product_start_date",
        "product_end_date",
        "date_source",
        "date_badge_text",
        "bbox",
    ]


def copy_bbox(box):
    if not valid_bbox(box):
        return None
    return [float(value) for value in box]


def serialize_bbox(box):
    normalized = copy_bbox(box)
    if normalized is None:
        return None
    return json.dumps(normalized)


def initialize_bbox_column(df):
    """Normalize bbox and discard obsolete duplicate bbox fields."""
    df = df.drop(columns=["ai_bbox", "corrected_bbox"], errors="ignore")

    if "bbox" not in df.columns:
        df["bbox"] = None

    for idx in df.index:
        df.at[idx, "bbox"] = copy_bbox(df.at[idx, "bbox"])

    return df


def normalized_bbox_to_pixels(box, image_size):
    normalized = copy_bbox(box)
    if normalized is None:
        return None

    image_width, image_height = image_size
    x1, y1, x2, y2 = normalized
    return [
        x1 / 1000 * image_width,
        y1 / 1000 * image_height,
        (x2 - x1) / 1000 * image_width,
        (y2 - y1) / 1000 * image_height,
    ]


def pixel_bbox_to_normalized(box, image_size):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None

    try:
        x, y, width, height = [float(value) for value in box]
    except (TypeError, ValueError):
        return None

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        return None

    normalized = [
        max(0.0, min(1000.0, x / image_width * 1000)),
        max(0.0, min(1000.0, y / image_height * 1000)),
        max(0.0, min(1000.0, (x + width) / image_width * 1000)),
        max(0.0, min(1000.0, (y + height) / image_height * 1000)),
    ]
    return copy_bbox(normalized)


def save_bbox_edit(row_index, box, input_keys, notice_key):
    normalized = copy_bbox(box)
    if normalized is None:
        return

    edited_df = st.session_state["edited_df"].copy()
    edited_df.at[row_index, "bbox"] = list(normalized)
    st.session_state["edited_df"] = edited_df

    for key, value in zip(input_keys, normalized):
        st.session_state[key] = float(value)

    st.session_state[notice_key] = True


def correction_text(value):
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def log_corrections(original_df, edited_df, review_id):
    changes = []
    now = datetime.now(timezone.utc).isoformat()

    for idx in edited_df.index:
        if idx not in original_df.index:
            continue

        for field in editable_columns():
            old = original_df.at[idx, field] if field in original_df.columns else None
            new = edited_df.at[idx, field] if field in edited_df.columns else None

            old_text = correction_text(old)
            new_text = correction_text(new)

            if old_text != new_text:
                changes.append(
                    {
                        "review_id": review_id,
                        "timestamp": now,
                        "page": edited_df.at[idx, "page"],
                        "product_name": edited_df.at[idx, "product_name"],
                        "field": field,
                        "ai_value": old_text,
                        "corrected_value": new_text,
                    }
                )

    if changes:
        write_header = not CORRECTIONS_PATH.exists()
        with CORRECTIONS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=changes[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(changes)

    return len(changes)


def save_review(result, edited_df, status):
    review_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reviewed_at = datetime.now(timezone.utc).isoformat()

    corrections = log_corrections(
        st.session_state["original_df"],
        edited_df,
        review_id,
    )

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO reviews (
            review_id, reviewed_at, status,
            shop_name, campaign_name, product_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            reviewed_at,
            status,
            result.get("shop_name"),
            result.get("campaign_name"),
            len(edited_df),
        ),
    )

    if status == "approved":
        for _, row in edited_df.iterrows():
            bbox = serialize_bbox(row.get("bbox"))

            conn.execute(
                """
                INSERT INTO approved_products (
                    review_id, reviewed_at,
                    shop_name, campaign_name, region, branch,
                    flyer_start_date, flyer_end_date, page,
                    product_name, quantity, price_before, price_after,
                    currency, product_start_date, product_end_date,
                    date_source, date_badge_text, bbox
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    reviewed_at,
                    row.get("shop_name"),
                    row.get("campaign_name"),
                    row.get("region"),
                    row.get("branch"),
                    row.get("flyer_start_date"),
                    row.get("flyer_end_date"),
                    int(row.get("page")),
                    row.get("product_name"),
                    row.get("quantity"),
                    row.get("price_before"),
                    row.get("price_after"),
                    row.get("currency"),
                    row.get("product_start_date"),
                    row.get("product_end_date"),
                    row.get("date_source"),
                    row.get("date_badge_text"),
                    bbox,
                ),
            )

    conn.commit()
    conn.close()
    return review_id, corrections


def prepare_result_from_json(pdf_path, work_dir, uploaded_json):
    """
    Fallback mode:
    use the JSON produced by the working Colab pipeline, then review it here.
    """
    data = json.load(uploaded_json)

    page_images = render_pdf(
        pdf_path=pdf_path,
        page_dir=Path(work_dir) / "pages",
        scale=1.3,
    )

    products = data.get("products", [])
    if not isinstance(products, list):
        raise ValueError("JSON must contain a 'products' list.")

    return {
        "shop_name": data.get("shop_name"),
        "campaign_name": data.get("campaign_name"),
        "flyer_start_date": data.get("flyer_start_date"),
        "flyer_end_date": data.get("flyer_end_date"),
        "region": data.get("region"),
        "branch": data.get("branch"),
        "currency": data.get("currency"),
        "products": products,
        "_page_images": page_images,
        "_usage": [],
        "_model": data.get("_model", "pre-extracted JSON"),
    }


init_database()

mode = st.sidebar.radio(
    "Mode",
    [
        "Live extraction with OpenRouter",
        "Review existing Colab JSON",
    ],
)

uploaded_pdf = st.file_uploader("Upload flyer PDF", type=["pdf"])

api_key = ""
uploaded_json = None

if mode == "Live extraction with OpenRouter":
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if api_key:
        st.sidebar.success("OpenRouter API key loaded from the environment.")
    else:
        st.sidebar.warning(
            "Set OPENROUTER_API_KEY before launching Streamlit to enable "
            "live extraction."
        )
else:
    uploaded_json = st.file_uploader(
        "Upload the JSON produced by your Colab pipeline",
        type=["json"],
    )

if uploaded_pdf is not None:
    st.write(f"**File:** {uploaded_pdf.name}")

    ready = (
        bool(api_key)
        if mode == "Live extraction with OpenRouter"
        else uploaded_json is not None
    )

    if st.button("Process flyer", type="primary", disabled=not ready):
        session_dir = tempfile.mkdtemp(prefix="flyer_ai_local_")
        pdf_path = Path(session_dir) / uploaded_pdf.name
        pdf_path.write_bytes(uploaded_pdf.getvalue())

        try:
            if mode == "Live extraction with OpenRouter":
                with st.spinner("Qwen3-VL-32B is reading the flyer..."):
                    result = process_flyer(
                        pdf_path=pdf_path,
                        work_dir=session_dir,
                        api_key=api_key,
                        model_id=MODEL_ID,
                    )
            else:
                result = prepare_result_from_json(
                    pdf_path=pdf_path,
                    work_dir=session_dir,
                    uploaded_json=uploaded_json,
                )

            st.session_state["flyer_result"] = result

            df = pd.DataFrame(result["products"])

            required = {
                "page": None,
                "product_name": None,
                "quantity": None,
                "price_before": None,
                "price_after": None,
                "currency": None,
                "product_start_date": None,
                "product_end_date": None,
                "date_source": None,
                "date_badge_text": None,
                "bbox": None,
            }

            for col, default in required.items():
                if col not in df.columns:
                    df[col] = default

            df = initialize_bbox_column(df)

            flyer_fields = [
                "shop_name",
                "campaign_name",
                "region",
                "branch",
                "flyer_start_date",
                "flyer_end_date",
            ]

            for field in flyer_fields:
                if field not in df.columns:
                    df[field] = result.get(field)

            st.session_state["original_df"] = df.copy(deep=True)
            st.session_state["edited_df"] = df.copy(deep=True)

            # A new flyer may reuse page/product indices from the previous one.
            # Clear only the manual bbox widgets so they receive fresh defaults.
            for key in list(st.session_state):
                if str(key).startswith("bbox_input_"):
                    del st.session_state[key]

            st.success("Ready for review.")

        except Exception as exc:
            st.exception(exc)


if "flyer_result" in st.session_state:
    result = st.session_state["flyer_result"]
    original_df = st.session_state["original_df"]

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Shop", result.get("shop_name") or "—")
    c2.metric("Products", len(original_df))
    c3.metric(
        "Flyer dates",
        f"{result.get('flyer_start_date') or '—'} → "
        f"{result.get('flyer_end_date') or '—'}",
    )

    page_values = original_df["page"].dropna()
    if len(page_values) == 0:
        st.warning("No page numbers were found in the extracted data.")
        st.stop()

    pages = sorted(page_values.astype(int).unique().tolist())
    page = st.selectbox("Page to review", pages)

    edited_source = st.session_state["edited_df"]
    page_df = edited_source[
        edited_source["page"].astype(int) == page
    ].copy()
    image_path = result["_page_images"][page - 1]

    left, right = st.columns([1.0, 1.3], gap="large")

    with right:
        st.subheader(f"Extracted products — Page {page}")
        st.caption("Edit any wrong value directly in the table.")

        display_cols = [
            "product_name",
            "quantity",
            "price_before",
            "price_after",
            "currency",
            "product_start_date",
            "product_end_date",
            "date_source",
            "date_badge_text",
            "bbox",
        ]

        editor = st.data_editor(
            page_df[display_cols],
            use_container_width=True,
            disabled=["bbox"],
            key=f"editor_{page}",
        )

        edited_full = st.session_state["edited_df"].copy()

        for idx in editor.index:
            for column in editor.columns:
                if column == "bbox":
                    continue
                edited_full.at[idx, column] = editor.at[idx, column]

        st.session_state["edited_df"] = edited_full

        if len(editor) == 0:
            st.info("No products on this page.")
            selected_idx = None
            candidate_bbox = None
            box_is_valid = False
            input_keys = []
        else:
            selected_idx = st.selectbox(
                "Select product to highlight",
                editor.index.tolist(),
                format_func=lambda idx: str(
                    editor.at[idx, "product_name"]
                    or f"Product {idx + 1}"
                ),
            )

            selected_row = st.session_state["edited_df"].loc[selected_idx]
            saved_bbox = copy_bbox(selected_row.get("bbox"))
            input_bbox = saved_bbox or [0.0, 0.0, 0.0, 0.0]

            st.caption("Correct bounding box (normalized coordinates, 0–1000)")
            box_cols = st.columns(4)
            coordinate_names = ("X1", "Y1", "X2", "Y2")
            coordinate_values = []
            input_keys = [
                f"bbox_input_{page}_{selected_idx}_{name.lower()}"
                for name in coordinate_names
            ]

            for position, (column, name, input_key) in enumerate(
                zip(box_cols, coordinate_names, input_keys)
            ):
                with column:
                    coordinate_values.append(
                        st.number_input(
                            name,
                            min_value=0.0,
                            max_value=1000.0,
                            value=float(input_bbox[position]),
                            step=1.0,
                            key=input_key,
                        )
                    )

            candidate_bbox = [float(value) for value in coordinate_values]
            box_is_valid = valid_bbox(candidate_bbox)

            if box_is_valid:
                st.caption("Number-input coordinates are valid.")
            else:
                st.error("Bounding box must satisfy X1 < X2 and Y1 < Y2.")

    with left:
        st.subheader(f"Flyer — Page {page}")
        canvas_bbox = None

        if selected_idx is None:
            st.image(image_path, use_container_width=True)
        else:
            initial_bbox = candidate_bbox if box_is_valid else saved_bbox

            if valid_bbox(initial_bbox):
                with Image.open(image_path) as flyer_image:
                    image_size = flyer_image.size

                pixel_bbox = normalized_bbox_to_pixels(
                    initial_bbox,
                    image_size,
                )
                bbox_token = "_".join(
                    f"{value:.2f}" for value in initial_bbox
                )
                canvas_result = detection(
                    image_path=image_path,
                    label_list=["Product"],
                    bboxes=[pixel_bbox],
                    labels=[0],
                    height=900,
                    width=700,
                    line_width=3,
                    use_space=False,
                    key=(
                        f"bbox_canvas_{page}_{selected_idx}_{bbox_token}"
                    ),
                )

                if canvas_result is not None:
                    if len(canvas_result) == 1:
                        canvas_bbox = pixel_bbox_to_normalized(
                            canvas_result[0].get("bbox"),
                            image_size,
                        )
                        if canvas_bbox is None:
                            st.error(
                                "The image editor returned an invalid box. "
                                "Use the number inputs instead."
                            )
                    else:
                        st.error(
                            "Keep exactly one rectangle in the image editor. "
                            "The extra rectangle was not accepted."
                        )

                st.caption(
                    "Click the rectangle to select it, then drag or resize it. "
                    "Click Complete inside the editor before Save Bounding Box."
                )
            else:
                st.image(image_path, use_container_width=True)
                st.info(
                    "This product has no valid box yet. Enter valid coordinates "
                    "with the number inputs, then save them."
                )

    if selected_idx is not None:
        pending_bbox = canvas_bbox or (
            candidate_bbox if box_is_valid else None
        )
        notice_key = f"bbox_saved_{page}_{selected_idx}"

        with right:
            if canvas_bbox is not None:
                rounded_canvas_bbox = [round(value, 2) for value in canvas_bbox]
                st.caption(f"Editor result: {rounded_canvas_bbox}")

            st.button(
                "Save Bounding Box",
                disabled=pending_bbox is None,
                key=f"save_bbox_{page}_{selected_idx}",
                on_click=save_bbox_edit,
                args=(selected_idx, pending_bbox, input_keys, notice_key),
            )

            if st.session_state.pop(notice_key, False):
                st.success("Bounding box saved for this product.")

    st.divider()

    edited_df = st.session_state["edited_df"]

    approve_col, reject_col, _ = st.columns([1, 1, 4])

    with approve_col:
        if st.button("Approve flyer", type="primary", use_container_width=True):
            review_id, correction_count = save_review(
                result,
                edited_df,
                "approved",
            )
            st.success(
                f"Approved and saved to SQLite. "
                f"{correction_count} correction(s) logged."
            )

    with reject_col:
        if st.button("Reject flyer", use_container_width=True):
            review_id, correction_count = save_review(
                result,
                edited_df,
                "rejected",
            )
            st.warning(
                f"Rejected. {correction_count} correction(s) logged."
            )

    with st.expander("Run details / downloads"):
        st.write("Model:", result.get("_model"))

        usage = pd.DataFrame(result.get("_usage", []))
        if len(usage):
            st.dataframe(usage, use_container_width=True, hide_index=True)

        st.download_button(
            "Download reviewed CSV",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="reviewed_flyer_products.csv",
            mime="text/csv",
        )
