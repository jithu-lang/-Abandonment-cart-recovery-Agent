
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

from test import build_cart_recovery_workflow
from test import CartRecoveryState

from Webhooks.shopify import router as shopify_router

app = FastAPI(
    title="AI Cart Recovery Agent API",
    description="Production API for LangGraph-powered Cart Recovery",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],       
    allow_headers=["*"],        
)
app.include_router(shopify_router)

workflow = build_cart_recovery_workflow()
class RecoveryRequest(BaseModel):
    cart_id: str
    cart_data: Dict
    discount_offers: List[Dict]
    skip_human_approval: bool = False   

class RecoveryResponse(BaseModel):
    cart_id: str
    selected_offer: str
    discount_amount: float
    email_subject: str
    email_body: str
    human_approved: bool
    dispatch_status: str
    recovery_probability: Optional[float]
    workflow_log: List[str]
    timestamp: str


@app.post("/recover", response_model=RecoveryResponse)
async def recover_cart(request: RecoveryRequest):

    initial_state: CartRecoveryState = {
        'cart_id': request.cart_id,
        'customer_data': {},
        'cart_data': request.cart_data,
        'discount_offers': request.discount_offers,
        'customer_segment': '',
        'cart_analysis': {},
        'abandonment_insights': {},
        'eligible_offers': [],
        'selected_offer': {},
        'profitability_analysis': {},
        'email_subject': '',
        'email_body': '',
        'human_approved': not request.skip_human_approval,  # auto-approve if skip=True
        'human_feedback': 'Auto-approved' if request.skip_human_approval else '',
        'dispatch_status': 'pending',
        'workflow_log': []
    }

    config = {"configurable": {"thread_id": request.cart_id}}
    
    final_state = None
    for state in workflow.stream(initial_state, config=config):
        final_state = state
   
    final_key = list(final_state.keys())[0]
    result = final_state[final_key]
    return RecoveryResponse(
        cart_id=request.cart_id,
        selected_offer=result['selected_offer'].get('offer_code', 'NONE'),
        discount_amount=result['selected_offer'].get('calculated_discount_amount', 0.0),
        email_subject=result.get('email_subject', ''),
        email_body=result.get('email_body', ''),
        human_approved=result.get('human_approved', False),
        dispatch_status=result.get('dispatch_status', 'pending'),
        recovery_probability=result['cart_analysis'].get('recovery_probability'),
        workflow_log=result['workflow_log'],
        timestamp=datetime.now().isoformat()
    )

@app.get("/")
def home():
    return {"message": "AI Cart Recovery Agent API is running!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
 