from fastapi import APIRouter, Request
import json

router = APIRouter(prefix="/webhooks", tags=["shopify"])

@router.post("/shopify")
async def shopify_cart_webhook(request: Request):
    body = await request.json()

    cart_id = body.get("id")

    cart_data = {
        "customer_name": body["customer"]["first_name"],
        "customer_email": body["customer"]["email"],
        "cart_value": float(body["total_price"]),
        "num_items": len(body["line_items"]),
        "products": body["line_items"],
        "time_since_abandonment_hours": 1
    }

    discount_offers = [] 

    # Triggers the workflow
    from test import build_cart_recovery_workflow
    workflow = build_cart_recovery_workflow()

    initial_state = {
        "cart_id": cart_id,
        "cart_data": cart_data,
        "discount_offers": discount_offers,
        "workflow_log": []
    }

    result = workflow.invoke(initial_state)

    return {"status": "ok", "workflow": result}
