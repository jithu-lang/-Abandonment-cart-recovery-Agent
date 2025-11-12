import streamlit as st
import pandas as pd
import json
import requests

API_URL = "http://127.0.0.1:8000/recover"

st.set_page_config(page_title="AI Cart Recovery Dashboard", layout="wide")
st.title("AI Cart Recovery Agent – Dashboard")
st.markdown("Select a cart, auto-generate JSON, and run the recovery agent.")
st.divider()
st.sidebar.header(" Load Dataset Files")

cart_file = st.sidebar.file_uploader("Upload CartData.csv", type=["csv"])
discount_file = st.sidebar.file_uploader("Upload DiscountData.csv", type=["csv"])

if cart_file and discount_file:
    cart_df = pd.read_csv(cart_file)
    discount_df = pd.read_csv(discount_file)

    st.success(" CSV files loaded successfully!")
else:
    st.warning("Upload both CartData.csv and DiscountData.csv to continue.")
    st.stop()

st.header(" Select a Cart")

cart_ids = cart_df["cart_id"].tolist()
selected_cart_id = st.selectbox("Choose a cart to process:", cart_ids)

selected_cart_row = cart_df[cart_df["cart_id"] == selected_cart_id].iloc[0]
cart_data = selected_cart_row.to_dict()
try:
    cart_data["products"] = json.loads(cart_data["products"])
except:
    try:
        cart_data["products"] = eval(cart_data["products"])
    except:
        cart_data["products"] = []

st.subheader("Cart Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Cart Value", f"₹{cart_data['cart_value']}")
col2.metric("Items", cart_data["num_items"])
col3.metric("Segment", cart_data["customer_segment"])

st.write("### Products in Cart")
st.json(cart_data["products"])
st.divider()
st.header(" Available Discount Offers")

discounts = discount_df.to_dict("records")
st.json(discounts)
st.divider()
st.header(" Auto-Generated JSON Preview")

auto_payload = {
    "cart_id": selected_cart_id,
    "cart_data": cart_data,
    "discount_offers": discounts,
    "skip_human_approval": True
}

st.code(json.dumps(auto_payload, indent=2), language="json")
st.divider()

if st.button(" Run Cart Recovery Agent"):
    with st.spinner("Running AI agent..."):
        response = requests.post(API_URL, json=auto_payload)

    if response.status_code == 200:
        res = response.json()
        st.success(" Agent completed!")

       
        colA, colB, colC = st.columns(3)
        colA.metric("Selected Offer", res["selected_offer"])
        colB.metric("Discount", f"₹{res['discount_amount']}")
        colC.metric("Recovery Probability", f"{res.get('recovery_probability', 0)}%")

        st.divider()
        st.subheader(" Email Subject")
        st.write(res["email_subject"])

        st.subheader(" Email HTML Preview")
        st.components.v1.html(res["email_body"], height=500, scrolling=True)

        st.divider()

        st.subheader(" Workflow Log")
        st.json(res["workflow_log"])

        st.info(f" Dispatch Status: {res['dispatch_status']}")
