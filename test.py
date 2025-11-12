import os
import pandas as pd
from typing import TypedDict, Dict, List
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import json
from Mail_sender.grid import mail_sender
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from ABtest.variant import ab_test
from ML.inference import predict_recovery_prob_advanced
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import re
import math


def cleaned_json(value, fallback=None):    
 if isinstance(value, (list, dict)):
        return value
 if isinstance(value, str):
        text = value.strip()
        text = re.sub(r"[\x00-\x1F\x7F]", "", text)
        text = (text.replace("“", "\""))
        text = (text.replace("”", "\""))
        text = (text.replace("‘", "'"))          
        text = (text.replace("’", "'"))
        try:
            return json.loads(text)
        except Exception:
            pass
        pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}|\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
             return json.loads(match.group(0))
            except Exception:
             pass
 return fallback if fallback is not None else []


groq_api_key = os.getenv('GROQ_API_KEY')
if  groq_api_key:

    llm = ChatGroq( model="llama-3.1-8b-instant", temperature=0.7, groq_api_key=groq_api_key, max_tokens=1024)
else :     print(f"ERROR: {str()}")
    

def email(template: str, context: dict) -> str:
 def change(match):
     expr = match.group(1).strip()
     if "|" in expr:
            var, filt = [x.strip() for x in expr.split("|", 1)]
            value = context.get(var, "")
            if filt == "currency":return f"${float(value):,.2f}"
            return str(value)
        
     return str(context.get(expr, ""))

 return re.sub(r"{{\s*(.*?)\s*}}", change, template)


class CartRecoveryState(TypedDict):
 cart_id: str
 customer_data: Dict
 cart_data: Dict
 discount_offers: List[Dict]
   
 customer_segment: str
 cart_analysis: Dict
 abandonment_insights: Dict
   
 eligible_offers: List[Dict]
 selected_offer: Dict
 profitability_analysis: Dict
 ab_variant: str
 mail_subject: str
 content: str
   
 approval: bool
 feedback: str
  
 dispatch_status: str
 workflow_log: List[str]

#gets the data from the cartdata that is from csv file
 
def load_datasets():
 try:
     cart_df = pd.read_csv('CartData.csv')
     discount_df = pd.read_csv('DiscountData.csv')
     return cart_df, discount_df
 except FileNotFoundError as e:
     print("\n ERROR: CSV files not found!")
     raise e
 
#gets the data according to the cart id

def get_cart(cart_id: str, cart_df: pd.DataFrame) -> Dict:
 cart_raw = cart_df[cart_df['cart_id'] == cart_id].iloc[0]
 return cart_raw.to_dict()

#the first node of the graph and the flow start from here

def cart_data(state: CartRecoveryState) -> CartRecoveryState: 
 time = f"[{datetime.now().strftime('%H:%M:%S')}] DATA INGESTION: Processing cart {state['cart_id']}"
 cart_raw = state['cart_data']
 products = cleaned_json(cart_raw.get('products', []), fallback=[])
 seg = cart_raw.get('customer_segment') or cart_raw.get('segment') or 'new'
 tone = cart_raw.get('customer_tone') or cart_raw.get('tone_preference') or 'casual'
 prev_purch = int(cart_raw.get('previous_purchases', 0))
 adds_last_month = int(cart_raw.get('cart_adds_last_month', cart_raw.get('engagement_score', 0)))
 device = cart_raw.get('device', 'mobile')
 geo = cart_raw.get('geo', 'IN')
 abandon_hours = int(cart_raw.get('time_since_abandonment_hours', 24))
 in_stock = bool(cart_raw.get('all_items_in_stock', True))
 price_sens = float(cart_raw.get('price_sensitivity', 0.5))
 intent_score = float(cart_raw.get('intent_score', 0.6))
 abandon_reason = cart_raw.get('abandonment_reason', 'unknown')
 cart_value = float(cart_raw.get('cart_value', 0.0))
 num_items = int(cart_raw.get('num_items', len(products)))
 clv = round((0.3 * cart_value) + (prev_purch * (cart_value * 0.15)) + (adds_last_month * 2.0), 2)
 churn_risk = min(1.0, round( (0.45 if seg == 'new' else 0.2) + (0.35 if abandon_hours > 48 else 0.1) + (0.35 * price_sens), 2))
 urgency = "high" if (not in_stock or abandon_hours > 24) else "medium"
 enriched_customer = {'customer_id': cart_raw.get('customer_id'),'name': cart_raw.get('customer_name', 'Customer'),'email': cart_raw.get('customer_email', ''),
        'segment': seg,
        'tone_preference': tone,
        'previous_purchases': prev_purch,
        'engagement_score': adds_last_month,
        'device': device,
        'geo': geo,
        'price_sensitivity': price_sens,
        'intent_score': intent_score,
        'clv': clv,
        'churn_risk': churn_risk}
 enriched_cart = {'cart_id': cart_raw.get('cart_id'),'cart_value': round(cart_value, 2), 'num_items': num_items, 'products': products, 'time_since_abandonment_hours': abandon_hours, 'all_items_in_stock': in_stock,
        'urgency': urgency,
        'abandonment_reason': abandon_reason}
 state['customer_data'] = enriched_customer
 state['cart_data'] = enriched_cart
 state['workflow_log'].append(time)
 print(f"{time}")
 return state


def analysis(state: CartRecoveryState) -> CartRecoveryState:
    
 time = f"[{datetime.now().strftime('%H:%M:%S')}] CUSTOMER ANALYSIS: Analyzing behavior patterns"
 cd = state['customer_data']
 cart = state['cart_data']
 variant =ab_test (cd['customer_id'])
 state['ab_variant'] = variant
    
 if variant == "A":
        cd['ab_persona'] = "value-seeking"
 else:
        cd['ab_persona'] = "premium-engager"

 primary_category = (cart['products'][0]['category']
 if cart['products'] else 'Unknown')
 analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a senior growth analyst for an e-commerce platform.\n"
            "Your ONLY task is to return a JSON object evaluating the customer and cart.\n"
            "VERY IMPORTANT RULES:\n"
            "1. You MUST return STRICT VALID JSON.\n"
            "2. Do NOT add ANY text before or after the JSON.\n"
            "3. Do NOT add explanations, markdown, or commentary.\n"
            "4. Do NOT wrap the JSON in code fences.\n"
            "5. Strings must NOT contain unescaped newlines.\n"
            "If you cannot produce valid JSON, return an empty object {}."
        )),
        HumanMessage(content=json.dumps({
                "customer": {
                "segment": cd['segment'],
                "previous_purchases": cd['previous_purchases'],
                "engagement_score": cd['engagement_score'],
                "device": cd['device'],
                "geo": cd['geo'],
                "price_sensitivity": cd['price_sensitivity'],
                "intent_score": cd['intent_score'],
                "clv": cd['clv'],
                "churn_risk": cd['churn_risk'],
            },
            "cart": {
                "value": cart['cart_value'],
                "num_items": cart['num_items'],
                "urgency": cart['urgency'],
                "time_since_abandonment_hours": cart['time_since_abandonment_hours'],
                "in_stock": cart['all_items_in_stock'],
                "abandonment_reason": cart['abandonment_reason'],
                "primary_category": primary_category,
                "products_sample": cart['products'][:3],
            },
            "return_format": {
                "customer_value_tier": "low|medium|high",
                "abandonment_risk_level": "low|medium|high",
                "recovery_probability": "0..100",
                "primary_product_category": "string",
                "recommended_urgency_level": "low|medium|high",
                "key_insights": ["string", "string"]
            }
        }, indent=2))
    ])


 try:
        resp = llm.invoke(analysis_prompt.format_messages())
        text = getattr(resp, "content", str(resp))
        cleaned = re.sub(r"[\x00-\x1F\x7F]", "", text)
        cleaned = (cleaned.replace("“", "\"")
                 .replace("”", "\"")
                 .replace("‘", "'")
                 .replace("’", "'"))
    
        pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        match = re.search(pattern, cleaned, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in LLM output")

        result = json.loads(match.group(0))
        advanced_features = {
            "cart_value": cart["cart_value"],
            "time_since_abandonment_hours": cart["time_since_abandonment_hours"],
            "num_items": cart["num_items"],
            "value_per_item": cart["cart_value"] / max(1, cart["num_items"]),
            "previous_purchases": cd["previous_purchases"],
            "cart_adds_last_month": cd["engagement_score"],
            "engagement_ratio": cd["engagement_score"] / (cd["previous_purchases"] + 1),
            "discount_sensitivity": cart["cart_value"] / (cd["previous_purchases"] + 2),
            "customer_segment": cd["segment"],
            "customer_tone": cd["tone_preference"],
            "all_items_in_stock": cart["all_items_in_stock"]
        }


        
        ml_prob = predict_recovery_prob_advanced(advanced_features)
        result["ml_recovery_probability"] = ml_prob
 except Exception as e:
        print(f" LLM analysis failed: {e}; using heuristic fallback")
        result = {
            "customer_value_tier": "high" if cd['clv'] > 800 else "medium",
            "abandonment_risk_level": "high" if cd['churn_risk'] > 0.6 else "medium",
            "recovery_probability": min(95, int(70 + (cd['intent_score'] - cd['price_sensitivity']) * 20)),
            "primary_product_category": primary_category,
            "recommended_urgency_level": cart['urgency'],
            "key_insights": [
                "Price sensitivity influences discount need",
                "Timely follow-up increases conversion"
            ]
        }
 state['cart_analysis'] = result
 state['customer_segment'] = cd['segment']
 state['workflow_log'].append(time)
 print(f" {time}  |  Recovery Probability:   {result.get('recovery_probability', 'N/A')}%")
 return state
#teh updated value in state will be parsed to this node for offer selection
def offer_selection(state: CartRecoveryState) -> CartRecoveryState:

 time = f"[{datetime.now().strftime('%H:%M:%S')}] OFFER SELECTION: Evaluating discount eligibility"
 cart = state['cart_data']
 cd = state['customer_data']
 ca = state['cart_analysis']
 offers = state['discount_offers']
 cart_value = cart['cart_value']
 seg = cd['segment']
 primary_cat = ca.get('primary_product_category', 'Unknown')
 price_sens = cd['price_sensitivity']
 churn = cd['churn_risk']
 clv = cd['clv']
 urgency = cart['urgency']
 variant = state.get("ab_variant", "A")
 ml_prob = float(ca.get("ml_recovery_probability", 0.5))
 eligible = []

 for offer in offers:
        
        if cart_value < float(offer.get('min_cart_value', 0)):
            continue


        eligible_segments = str(offer.get('eligible_segments', 'all')).split(',')
        if 'all' not in eligible_segments and seg not in eligible_segments:
            continue


        eligible_categories = offer.get('eligible_categories', 'all')
        if eligible_categories != 'all' and primary_cat not in str(eligible_categories):
            continue
        dtype = offer.get('discount_type', 'percentage')
        dval = float(offer.get('discount_value', 0))
        max_amt = float(offer.get('max_discount_amount', dval))
        if dtype == 'percentage':
            disc = min(cart_value * (dval / 100.0), max_amt)
        else:
            disc = min(dval, cart_value)
        final_val = max(0.0, cart_value - disc)
        gross_margin = 0.40 * cart_value
        net_margin = gross_margin - disc
        profit_score = (net_margin / max(1.0, cart_value)) * 100.0
        roi_priority = float(offer.get('roi_priority', 0.6))
        business_score = (
            0.45 * (profit_score / 100.0) +
            0.20 * (min(clv, 1500) / 1500.0) +
            0.15 * (1.0 - churn) +
            0.10 * roi_priority +
            0.10 * (1.0 if urgency == "high" else 0.6 if urgency == "medium" else 0.3)
        )

        business_score += 0.05 * price_sens * (disc / max(1.0, cart_value))
        
        if ml_prob < 0.40:  
            business_score += 0.10 * (disc / cart_value)  # this will gieve big discounts
        elif ml_prob > 0.75:
            business_score += 0.05 * (profit_score / 100.0)  #this will give high margins
        
        if variant == "A":
            
            business_score += 0.07 * (profit_score / 100.0)
        else:
            
            business_score += 0.07 * (disc / cart_value)


        o = offer.copy()
        o.update({ "calculated_discount_amount": round(disc, 2), "final_cart_value": round(final_val, 2), "net_margin": round(net_margin, 2), "profitability_score": round(profit_score, 2),
            "offer_score": round(business_score, 4),
            "variant_used": variant,
            "ml_recovery_probability": ml_prob
        })
        eligible.append(o)


 eligible.sort(key=lambda x: x["offer_score"], reverse=True)


 if eligible:
        selected = eligible[0]
 else:
        selected = {"offer_code": "REMINDER_ONLY","discount_type": "none","discount_value": 0,"calculated_discount_amount": 0.0,"final_cart_value": cart_value,"profitability_score": (0.40 * cart_value) / max(1.0, cart_value) * 100,
            "offer_score": 0.2,
            "description": "Reminder only; no eligible profitable offers"
        }
        eligible = [selected]


 state['eligible_offers'] = eligible
 state['selected_offer'] = selected
 state['profitability_analysis'] = {
        "num_eligible_offers": len(eligible),
        "selected_offer_code": selected["offer_code"],
        "discount_amount": selected["calculated_discount_amount"],
        "final_value": selected["final_cart_value"],
        "profitability_score": selected["profitability_score"],
        "offer_score": selected["offer_score"],
    }
 state['workflow_log'].append(time)
 print(f" {time} | Selected: {selected['offer_code']} | OfferScore: {selected['offer_score']}")
 return state
#the llm invokes the email based on the tone of the customer
def mail(state: CartRecoveryState) -> CartRecoveryState:

 time = f"[{datetime.now().strftime('%H:%M:%S')}] EMAIL GENERATION: Creating personalized message"


 cd = state['customer_data']
 cart = state['cart_data']
 sel = state['selected_offer']
 ca = state['cart_analysis']
 tone_map = { 'casual': 'friendly and conversational', 'professional': 'polite and professional', 'enthusiastic': 'energetic and exciting'}
 tone = tone_map.get(cd['tone_preference'], 'friendly')
 reason = cart.get('abandonment_reason', 'unknown')
 reassurance = {
        "payment_failure": "We’ve improved payment reliability—your preferred method should work now.",
        "comparison_shopping": "This is one of our best prices right now—your items are still reserved.",
        "high_shipping_cost": "We’ve applied savings to offset your delivery cost.",
        "long_checkout_process": "Your cart is one click away—no extra forms this time.",
        "site_crash": "Sorry for the hiccup—everything’s stable now.", }.get(reason, "")

#we give the commands to the model here 
 email_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            f"You are an e-commerce email copywriter. Write a {tone} cart recovery email."
             "Write a personalized cart recovery email.\n\n"
             "STRICT RULES YOU MUST FOLLOW:\n"
             "1. Output ONLY a JSON object.\n"
             "2. JSON must contain exactly TWO keys: `subject_line` and `content`.\n"
             "3. `content` must be valid HTML wrapped in a single string.\n"
             "4. DO NOT include newlines that break JSON formatting.\n"
             "5. DO NOT include markdown, commentary, or extra text.\n"
             "6. DO NOT say anything outside the JSON object.\n"
            "7. If unsure, return an empty JSON object {}.\n" )),
        HumanMessage(content=json.dumps({
            "customer_name": cd['name'],
            "segment": cd['segment'],
            "cart_value": cart['cart_value'],
            "num_items": cart['num_items'],
            "selected_offer": {
            "code": sel.get('offer_code', 'REMINDER_ONLY'),
            "discount_amount": sel.get('calculated_discount_amount', 0.0),
            "type": sel.get('discount_type', 'none')},
            "urgency": ca.get("recommended_urgency_level", cart["urgency"]),
            "abandonment_reason": reason,
            "device": cd["device"],
            "geo": cd["geo"],
            "reassurance": reassurance,
            "constraints": {
            "subject_max_chars": 60,
            "cta_text": "Complete My Order",
            "include_code_if_available": True } }, indent=2))])


 try:
        resp = llm.invoke(email_prompt.format_messages())
        text = getattr(resp, "content", str(resp))
        cleaned = re.sub(r"[\x00-\x1F\x7F]", "", text)   
        cleaned = cleaned.replace("“", "\"")
        cleaned = cleaned.replace("”", "\"")
        cleaned = cleaned.replace("‘", "'")
        cleaned = cleaned.replace("’", "'")
        cleaned = cleaned.replace("\n", "\\n")
        cleaned = cleaned.replace("\r", "")
        cleaned = cleaned.replace("\t", " ")
        cleaned = cleaned.replace("\\", "\\\\")

        pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        match = re.search(pattern, cleaned, re.DOTALL)
        if not match:
          raise ValueError("No valid JSON found")
        data = json.loads(match.group(0))
        subject_line = data.get("subject_line") or f"Finish your order—save ${int(sel.get('calculated_discount_amount', 0))}"
        content = data.get("content")
        if not content:
            raise ValueError("missing body")
 except Exception as e:
        print(f" Email generation failed: {e}; using fallback template")
        subject_line = f"Complete your order and save ${int(sel.get('calculated_discount_amount', 0))}" if sel.get('calculated_discount_amount', 0) else "Your cart is still saved"
        code_html = f"Use code <b>{sel.get('offer_code')}</b> at checkout." if sel.get('discount_type') != 'none' else "Resume checkout in one click."
        content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 620px; margin:0 auto; padding:24px;">
          <h2 style="margin:0 0 12px;color:#222;">Hi {cd['name']},</h2>
          <p>We saved your cart with {cart['num_items']} item(s) worth <b>${cart['cart_value']}</b>.</p>
          {f"<p>{reassurance}</p>" if reassurance else ""}
          <p>{code_html}</p>
          <p><a href="#" style="background:#2563eb;color:#fff;padding:12px 18px;text-decoration:none;border-radius:6px;display:inline-block;">Complete My Order</a></p>
          <p style="color:#6b7280;font-size:13px;">If you need help, just reply to this email.</p>
        </body>
        </html>
        """


 state['mail_subject'] = subject_line
 state['content'] = content

 context = {
    "customer_name": state['customer_data']['name'],
    "num_items": state['cart_data']['num_items'],
    "cart_value": state['cart_data']['cart_value'],
    "selected_offer": state['selected_offer']['offer_code'],
    "discount_amount": state['selected_offer'].get("calculated_discount_amount", 0),
    "reassurance": "Everything looks good now.",
    "complete_order_url": f"https://mystore.com/recover/{state['cart_id']}"}


 final_email = email(state['content'], 
                    context)
 state['content'] = final_email

 state['workflow_log'].append(time)
 print(f" {time} | Subject: {subject_line}")
 return state
#view the basic analysis of the customer analysus
def reviewer(state: CartRecoveryState) -> CartRecoveryState:
 cd, cart, sel, ca = state['customer_data'], state['cart_data'], state['selected_offer'], state['cart_analysis']
 brief = (
        f"- Segment: {cd['segment']} | CLV: ${cd['clv']} | Churn risk: {cd['churn_risk']}\n"
        f"- Cart: ${cart['cart_value']} • {cart['num_items']} items • Urgency: {cart['urgency']}\n"
        f"- Reason: {cart.get('abandonment_reason','unknown')} • Device: {cd['device']} • Geo: {cd['geo']}\n"
        f"- Offer: {sel['offer_code']} → ${sel.get('calculated_discount_amount',0)} off "
        f"(ProfitScore: {state['profitability_analysis'].get('profitability_score','-')} | "
        f"OfferScore: {state['profitability_analysis'].get('offer_score','-')})\n"
        f"- LLM Recovery Probability: {ca.get('recovery_probability','N/A')}%" )
 print("\nREVIEWER BRIEF\n" +  f"\n{brief}\n")
 state['workflow_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] REVIEWER BRIEF shown")
 return state
#reviewer  approves for the email

def approval(state: CartRecoveryState) -> CartRecoveryState:

 time = f"[{datetime.now().strftime('%H:%M:%S')}] HUMAN APPROVAL: Waiting for review" 
 print("APPROVAL REQUIRED")
 print(f"\nCart ID: {state['cart_id']}")
 print(f"Customer: {state['customer_data']['name']} ({state['customer_data']['email']})")
 print(f"Cart Value: ${state['cart_data']['cart_value']}")
 print(f"Selected Offer: {state['selected_offer']['offer_code']}")
 print(f"Discount Amount: ${state['selected_offer']['calculated_discount_amount']}")
 print(f"\n Email Subject: {state['mail_subject']}")
 print(f"\n Email Preview:")
 preview = state['content'][:500] + "..." if len(state['content']) > 500 else state['content']
 print(preview)
    
 approval = input("\n Approve this email? (yes/no): ").strip().lower()
   
 if approval == 'yes':
        state['approval'] = True
        state['feedback'] = "Approved for dispatch"
        print("Email APPROVED for dispatch")
 else:
        state['approval'] = False
        feedback = input(" Reason for rejection: ").strip()
        state['feedback'] = feedback or "Rejected by reviewer"
        print("Email REJECTED")
   
 state['workflow_log'].append(time)
 return state

#dispatching of the node happens here
def dispatch(state):
 to = state['customer_data']['email']
 subject = state['mail_subject']
 html = state['content']
 status = mail_sender(to, subject, html)
 state['dispatch_status'] = 'sent' if status == 202 else 'failed'
 state['workflow_log'].append(f" Email dispatched via SendGrid ({status})")
 return state
#conditional edges chosses the right node to functionn
def should_dispatch(state: CartRecoveryState) -> str:
    if state['approval']:
        return "dispatch"
    else:
        return "end"
#the workflow of the graph happens here
def graph_workflow():
 workflow = StateGraph(CartRecoveryState)
 workflow.add_node("cart_data", cart_data)
 workflow.add_node("cart_analysis", analysis)
 workflow.add_node("offer_selection", offer_selection)
 workflow.add_node("mail_generation", mail)
 workflow.add_node("reviewer_brief", reviewer)
 workflow.add_node("approvala_state", approval)
 workflow.add_node("dispatch", dispatch)
 workflow.set_entry_point("cart_data")
 workflow.add_edge("cart_data", "cart_analysis")
 workflow.add_edge("cart_analysis", "offer_selection")
 workflow.add_edge("offer_selection", "mail_generation")
 workflow.add_edge("mail_generation", "reviewer_brief")    
 workflow.add_edge("reviewer_brief", "approvala_state")
 workflow.add_conditional_edges( "approvala_state", should_dispatch, {"dispatch": "dispatch","end": END } )
 workflow.add_edge("dispatch", END)
   
 memory = MemorySaver()
 app = workflow.compile(checkpointer=memory)
   
 return app



def main(): 
 cart_df, discount_df = load_datasets()
 sample_cart_id = cart_df.iloc[0]['cart_id']
 cart_data = get_cart(sample_cart_id, cart_df)
   
 discount_offers = discount_df.to_dict('records')
 initial_state = {'cart_id': sample_cart_id,'customer_data': {},'cart_data': cart_data,'discount_offers': discount_offers,'customer_segment': '',
 'cart_analysis': {},
 'abandonment_insights': {},
 'eligible_offers': [],
 'selected_offer': {},
 'profitability_analysis': {},
 'mail_subject': '',
 'content': '',
 'approval': False,
 'feedback': '',
 'dispatch_status': 'pending',
 'workflow_log': [] } 
 app = graph_workflow()
 config = {"configurable": {"thread_id": sample_cart_id}}   
 final_state = None
 for state in app.stream(initial_state, config):
        final_state = state
    
 final_key = list(final_state.keys())[0]
 result = final_state[final_key]
 print(f"\nFinal Status: {result['dispatch_status']}")
 print(f"Human Decision: {result['feedback']}")
 print("\n Workflow Log:")
 for log in result['workflow_log']:
        print(f"  {log}") 
 output = {
    'cart_id': sample_cart_id,
    'customer_email': result['customer_data']['email'],
    'segment': result['customer_data']['segment'],
    'clv': result['customer_data']['clv'],
    'churn_risk': result['customer_data']['churn_risk'],
    'price_sensitivity': result['customer_data']['price_sensitivity'],
    'cart_value': result['cart_data']['cart_value'],
    'num_items': result['cart_data']['num_items'],
    'abandonment_reason': result['cart_data'].get('abandonment_reason', 'unknown'),
    'urgency': result['cart_data']['urgency'],
    'selected_offer': result['selected_offer']['offer_code'],
    'discount_amount': result['selected_offer'].get('calculated_discount_amount', 0.0),
    'profitability_score': result['profitability_analysis'].get('profitability_score', None),
    'offer_score': result['profitability_analysis'].get('offer_score', None),
    'estimated_recovery_probability': result['cart_analysis'].get('recovery_probability', None),
    'approval': result['approval'],
    'dispatch_status': result['dispatch_status'],
    'timestamp': datetime.now().isoformat()




    }
   
 output_df = pd.DataFrame([output])
 output_df.to_csv('recovery_output.csv', mode='a', header=not os.path.exists('recovery_output.csv'), index=False)

if __name__ == "__main__":
    main()



