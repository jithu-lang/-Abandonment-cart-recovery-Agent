import pandas as pd
def ab_test(customer_id: str):
 df = pd.read_csv(r"C:\Users\Vishnupriya\Downloads\jithu\ML project\Agentic Ai\CartData.csv")
 row = df[df["customer_id"] == customer_id]
 if row.empty:
     raise ValueError(f"Customer ID {customer_id} not found in dataset")
 row = row.iloc[0] 
 cart_value = float(row["cart_value"])
 price_sensitivity = float(row["price_sensitivity"])
 intent_score = float(row["intent_score"])
 abandonment_reason = str(row["abandonment_reason"])
 previous_purchases = int(row["previous_purchases"])
 price_related_reasons = ["price_too_high", "comparison_shopping", "high_shipping_cost"]
 if (cart_value < 800 or  price_sensitivity > 0.55 or intent_score < 0.55 or abandonment_reason in price_related_reasons or previous_purchases < 2):
        return "A"
 if (cart_value > 1500 or  price_sensitivity < 0.40 or intent_score > 0.65 or abandonment_reason not in price_related_reasons or previous_purchases >= 3):
        return "B"
 premium_score = 0
 value_score = 0
 if cart_value > 1200: premium_score += 1
 if price_sensitivity < 0.45: premium_score += 1
 if intent_score > 0.6: premium_score += 1
 if cart_value < 600: value_score += 1
 if price_sensitivity > 0.5: value_score += 1
 return "B" if premium_score > value_score else "A"
