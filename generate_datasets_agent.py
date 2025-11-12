
import random
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

customer_segments = ['new', 'returning', 'vip']
device_types = ['mobile', 'desktop', 'tablet']
geos = ['IN', 'UK', 'US', 'NZ']
abandon_reasons = [
    "high_shipping_cost",
    "payment_failure",
    "comparison_shopping",
    "site_crash",
    "long_checkout_process",
    "price_too_high",
    "distracted",]
product_categories = {
    "Electronics": (300, 1200),
    "Fashion": (20, 150),
    "Home": (30, 300),
    "Beauty": (10, 80),
    "Sports": (25, 400),
    "Books": (5, 40),}
customer_tones = ['casual', 'professional', 'enthusiastic']
print("Generated CartData.csv...")

cart_data = []

for i in range(1000):
    customer_id = f"CUST{1000 + i}"
    cart_id = f"CART{5000 + i}"
    segment = np.random.choice(customer_segments, p=[0.4, 0.45, 0.15])
    email = f"{customer_id.lower()}@example.com"
    phone = f"+91{random.randint(6000000000, 9999999999)}"
    name = f"Customer {i+1}"
    tone = np.random.choice(customer_tones)
    device = np.random.choice(device_types, p=[0.6, 0.35, 0.05])
    geo = np.random.choice(geos)
    num_items = np.random.randint(1, 6)
    products = []
    total_value = 0
    
    for j in range(num_items):
        category = np.random.choice(list(product_categories.keys()))
        price_min, price_max = product_categories[category]
        price = round(np.random.uniform(price_min, price_max), 2)
        product_name = f"{category} Item {j+1}"
        quantity = np.random.randint(1, 4)
        total_value += price * quantity
        
        products.append({
            'product_id': f"PROD{2000 + i*10 + j}",
            'product_name': product_name,
            'category': category,
            'price': float(price),
            'quantity': quantity
        })
    
    
    time_since_abandonment = np.random.randint(1, 72)  # hours
    previous_purchases = np.random.randint(0, 15) if segment != 'new' else 0
    price_sensitivity_score = round(np.random.uniform(0, 1), 2)
    intent_score = round(np.random.uniform(0.3, 0.9), 2)
    abandonment_reason = np.random.choice(abandon_reasons)
    cart_adds_last_month = np.random.randint(1, 20)
    in_stock = np.random.choice([True, False], p=[0.9, 0.1])
    
    cart_data.append({
        'cart_id': cart_id,
        'customer_id': customer_id,
        'customer_name': name,
        'customer_email': email,
        'customer_phone': phone,
        'customer_segment': segment,
        'customer_tone': tone,
        'device': device,
        'geo': geo,
        'cart_value': round(total_value, 2),
        'num_items': num_items,
        'products': json.dumps(products), 
        'time_since_abandonment_hours': time_since_abandonment,
        'previous_purchases': previous_purchases,
        'price_sensitivity': price_sensitivity_score,
        'intent_score': intent_score,
        'abandonment_reason': abandonment_reason,
        'cart_adds_last_month': cart_adds_last_month,
        'all_items_in_stock': in_stock,
        'abandonment_timestamp': (datetime.now() - timedelta(hours=time_since_abandonment)).isoformat()
    })


cart_df = pd.DataFrame(cart_data)
cart_df.to_csv('CartData.csv', index=False)
print(f" CartData.csv created with {len(cart_df)} records")
print(f"   Sample cart value range: ${cart_df['cart_value'].min():.2f} - ${cart_df['cart_value'].max():.2f}")
print()
print("Generated DiscountData.csv...")

discount_data = [
    {
        'offer_code': 'WELCOME10',
        'discount_type': 'percentage',
        'discount_value': 10,
        'min_cart_value': 50,
        'max_discount_amount': 50,
        'eligible_segments': 'new',
        'eligible_categories': 'all',
        'valid_from': '2025-01-01',
        'valid_until': '2025-12-31',
        'usage_limit_per_customer': 1,
        "roi_priority": 0.8,
        'description': 'Welcome offer for new customers'
    },
    {
        'offer_code': 'SAVE15',
        'discount_type': 'percentage',
        'discount_value': 15,
        'min_cart_value': 100,
        'max_discount_amount': 100,
        'eligible_segments': 'returning,vip',
        'eligible_categories': 'all',
        'valid_from': '2025-01-01',
        'valid_until': '2025-12-31',
        'usage_limit_per_customer': 5,
        "roi_priority": 0.6,
        'description': 'Loyalty discount for returning customers'
    },
    {
        'offer_code': 'VIP20',
        'discount_type': 'percentage',
        'discount_value': 20,
        'min_cart_value': 200,
        'max_discount_amount': 200,
        'eligible_segments': 'vip',
        'eligible_categories': 'all',
        'valid_from': '2025-01-01',
        'valid_until': '2025-12-31',
        'usage_limit_per_customer': 10,
        "roi_priority": 0.9,
        'description': 'VIP exclusive discount'
    },
    {
        'offer_code': 'FREESHIP',
        'discount_type': 'flat',
        'discount_value': 10,
        'min_cart_value': 75,
        'max_discount_amount': 10,
        'eligible_segments': 'all',
        'eligible_categories': 'all',
        'valid_from': '2025-01-01',
        'valid_until': '2025-12-31',
        'usage_limit_per_customer': 999,
        'roi_priority': 0.5,
        'description': 'Free standard shipping'
    },
    {
        'offer_code': 'FASHION25',
        'discount_type': 'percentage',
        'discount_value': 25,
        'min_cart_value': 150,
        'max_discount_amount': 150,
        'eligible_segments': 'all',
        'eligible_categories': 'Fashion',
        'valid_from': '2025-01-01',
        'valid_until': '2025-12-31',
        'usage_limit_per_customer': 3,
        'roi_priority': 0.7,
        'description': 'Fashion category special offer'
    },
    {
        'offer_code': 'ELECTRONICS30',
        'discount_type': 'flat',
        'discount_value': 30,
        'min_cart_value': 300,
        'max_discount_amount': 30,
        'eligible_segments': 'all',
        'eligible_categories': 'Electronics',
        'valid_from': '2025-01-01',
        'valid_until': '2025-12-31',
        'usage_limit_per_customer': 2,
        'roi_priority': 0.4,
        'description': 'Flat discount on electronics'
    },
    {
        'offer_code': 'LASTCHANCE',
        'discount_type': 'percentage',
        'discount_value': 12,
        'min_cart_value': 60,
        'max_discount_amount': 80,
        'eligible_segments': 'all',
        'eligible_categories': 'all',
        'valid_from': '2025-01-01',
        'valid_until': '2025-12-31',
        'usage_limit_per_customer': 999,
        'roi_priority': 0.5,
        'description': 'Last chance to complete your order'
    }
]


discount_df = pd.DataFrame(discount_data)
discount_df.to_csv('DiscountData.csv', index=False)
print(f"DiscountData.csv created with {len(discount_df)} offers")
print()
print(" ALL DATASETS GENERATED SUCCESSFULLY!")
print()
print(" Files created in current directory:")
print("   1. CartData.csv (100 abandoned cart records)")
print("   2. DiscountData.csv (7 discount offer codes)")
print()
print(" Quick Stats:")
print(f"   • Customer segments: {', '.join(customer_segments)}")
print(f"   • Product categories: {', '.join(product_categories)}")
print(f"   • Total cart value: ${cart_df['cart_value'].sum():,.2f}")
print()

