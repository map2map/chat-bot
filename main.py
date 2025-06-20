from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import asyncio
import os
from typing import Optional

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a tiny model for Render's free tier (under 300MB RAM)
model_name = "sshleifer/distilbart-cnn-12-6"  # Very small distilled BART model

# Initialize tokenizer and model with minimal memory footprint
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model with minimal memory usage...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for maximum compatibility
        low_cpu_mem_usage=True      # Keep memory usage as low as possible
    )
    model.eval()
    print(f"{model_name} loaded successfully (small model for free tier)")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e  # Re-raise the exception to see the full error

MAP2MAP_KNOWLEDGE = {
    "services": [
        "Business Listing Management - Regular updates and optimizations for your Google Business Profile",
        "NAP (Name, Address, Phone) Corrections - Keeping your business information accurate across the web",
        "Google Reviews Management - Professional responses to customer reviews",
        "Monthly Performance Reports - Detailed analytics on your Google Business Profile performance",
        "Local SEO Optimization - Improving your business visibility in local searches"
    ],
    "benefits": [
        "Improved local search rankings on Google",
        "Increased online visibility and customer reach",
        "Professional handling of customer reviews and reputation",
        "Accurate and consistent business information across the web",
        "Time-saving automated reporting and analytics"
    ],
    "pricing": "Our pricing is customized based on your business needs and the specific services required. We offer flexible packages for businesses of all sizes. Please contact our sales team at sales@map2map.com or visit https://map2map.com/pricing.html for a personalized quote.",
    "contact": "Email: sales@map2map.com | Phone: +919538886568 / +919538351398 | Website: https://map2map.com | Office Hours: Monday to Friday, 9:00 AM - 6:00 PM IST",
    "about": "Map2Map is a leading provider of Google Maps and Google Business Profile management services. We help businesses enhance their online presence, improve local search rankings, and effectively manage their digital reputation through professional and strategic optimization of their Google Business Profiles.",
    "timing": {
        "standard": "Most services show initial results within 1-2 days, with full optimization taking 2-3 weeks. Note that actual timing may vary depending on Google's review and approval process.",
        "listing_updates": "Business listing updates typically take 1-3 business days to reflect across platforms, subject to Google's approval process.",
        "review_responses": "We aim to respond to all new Google reviews within 24 hours during business days.",
        "reporting": "Monthly performance reports are delivered by the 5th of each month.",
        "support_response": "Our support team typically responds to inquiries within 1 business day."
    },
    "faq": {
        "claim_business": "To claim your business on Google, you'll need to verify your business through Google Business Profile. We can guide you through this process or handle it for you.",
        "improve_ranking": "Improving your Google Maps ranking involves several key strategies that we specialize in:\n\n1. **Complete Business Profile Optimization**: We ensure your Google Business Profile is 100% complete with accurate business information, relevant categories, high-quality photos, and detailed business descriptions.\n\n2. **Review Management**: We help you gather more positive reviews and professionally respond to all reviews, which significantly impacts your local search ranking.\n\n3. **NAP Consistency**: We maintain consistent Name, Address, and Phone number (NAP) information across all online directories and citations.\n\n4. **Local Citations & Backlinks**: We build high-quality local citations and backlinks from reputable directories and local business associations.\n\n5. **Google Posts & Q&A**: We regularly create engaging Google Posts and manage Q&A sections to improve engagement.\n\n6. **Local SEO Content**: We optimize your website content with local keywords and create location-specific landing pages.\n\n7. **Mobile Optimization**: We ensure your website is mobile-friendly, as Google prioritizes mobile-optimized sites in local search results.\n\nMost businesses see noticeable improvements within 2-3 months of consistent optimization. Would you like us to conduct a free audit of your current Google Business Profile?",
        "negative_reviews": "We help manage negative reviews by professionally responding to them and working with you to address any legitimate concerns raised by customers.",
        "service_area": "Yes, we serve businesses across India and can help optimize your Google Business Profile for any location.",
        "start_process": "To get started, simply contact our team with your business details and the services you're interested in. We'll provide a free consultation and quote.",
        "cancellation": "You can cancel or modify your service at any time. We offer flexible plans with no long-term contracts.",
        "data_security": "We take data security seriously and follow strict protocols to protect your business information. We only request access to the necessary accounts with your permission.",
        "difference": "Unlike many providers, we offer personalized service, detailed monthly reporting, and a dedicated account manager to ensure your business gets the attention it deserves."
    }
}

SYSTEM_PROMPT = f"""
You are Map2Map Assistant, an AI that provides information about Map2Map's Google Maps management services.

MAP2MAP KNOWLEDGE BASE:

SERVICES:
{chr(10).join(['- ' + s for s in MAP2MAP_KNOWLEDGE['services']])}

PRICING:
{MAP2MAP_KNOWLEDGE['pricing']}

CONTACT:
{MAP2MAP_KNOWLEDGE['contact']}

GUIDELINES:
1. Only provide information from the knowledge base above
2. If asked about pricing, direct to contact sales
3. Keep responses concise and professional
4. If you don't know the answer, direct to the website or support
5. Never make up information not in the knowledge base
"""

class Query(BaseModel):
    message: str

from fastapi import HTTPException
import traceback

@app.post("/chat")
async def chat(query: Query):
    try:
        print(f"\n--- New Chat Request ---")
        print(f"Message: {query.message}")
        
        # Handle common questions directly from our knowledge base first
        message_lower = query.message.lower()
        faq = MAP2MAP_KNOWLEDGE['faq']
        reply = None
        
        # Handle greetings
        if any(term in message_lower for term in ['hi', 'hello', 'hey', 'greetings']):
            reply = "Hello! I'm your Map2Map assistant. How can I help you with your Google Maps management today?"
            
        # FAQ Handling - Check these before model generation
        elif any(term in message_lower for term in ['claim', 'verify', 'verification', 'how do i claim']):
            reply = faq['claim_business']
        elif any(term in message_lower for term in ['rank', 'ranking', 'seo', 'search ranking', 'improve ranking']):
            reply = faq['improve_ranking']
        elif any(term in message_lower for term in ['negative', 'bad review', 'complaint']):
            reply = faq['negative_reviews']
        elif any(term in message_lower for term in ['location', 'area', 'city', 'country', 'service area']):
            reply = faq['service_area']
        elif any(term in message_lower for term in ['start', 'begin', 'get started', 'sign up', 'how do i start']):
            reply = faq['start_process']
        elif any(term in message_lower for term in ['cancel', 'stop', 'end service', 'cancel service']):
            reply = faq['cancellation']
        elif any(term in message_lower for term in ['security', 'privacy', 'data']):
            reply = faq['data_security']
        elif any(term in message_lower for term in ['different', 'better', 'unique', 'why choose', 'what makes you different']):
            reply = faq['difference']
        # Handle other common questions
        elif any(term in message_lower for term in ['price', 'cost', 'how much', 'payment']):
            reply = MAP2MAP_KNOWLEDGE['pricing']
        elif any(term in message_lower for term in ['contact', 'email', 'phone', 'reach', 'get in touch', 'call', 'number']):
            reply = f"You can reach us at:\n{MAP2MAP_KNOWLEDGE['contact']}"
        elif any(term in message_lower for term in ['service', 'what do you', 'offer', 'provide']):
            reply = "Map2Map offers the following services:\n" + "\n".join([f"• {s}" for s in MAP2MAP_KNOWLEDGE['services']])
        elif any(term in message_lower for term in ['what is map2map', 'who are you', 'about']):
            reply = f"{MAP2MAP_KNOWLEDGE['about']}\n\nOur services include:\n" + "\n".join([f"• {s}" for s in MAP2MAP_KNOWLEDGE['services']])
        elif any(term in message_lower for term in ['how long', 'when will', 'time frame', 'duration', 'when can i expect']):
            timing = MAP2MAP_KNOWLEDGE['timing']
            if 'listing' in message_lower or 'update' in message_lower:
                reply = timing['listing_updates']
            elif 'review' in message_lower or 'response' in message_lower:
                reply = timing['review_responses']
            elif 'report' in message_lower or 'analytics' in message_lower:
                reply = timing['reporting']
            elif 'support' in message_lower or 'response time' in message_lower:
                reply = timing['support_response']
            else:
                reply = f"{timing['standard']}\n\nMore specific timing information:\n" + \
                       f"\n• {timing['listing_updates']}" + \
                       f"\n• {timing['review_responses']}" + \
                       f"\n• {timing['reporting']}"
        elif any(term in message_lower for term in ['benefit', 'advantage', 'why should i', 'why choose']):
            reply = "Map2Map offers several benefits for your business:\n" + "\n".join([f"• {b}" for b in MAP2MAP_KNOWLEDGE['benefits']])
        
        # If no FAQ match, use the model to generate a response
        if reply is None:
            # Prepare the prompt with system message and user input
            prompt = f"""{SYSTEM_PROMPT}
            
            Current conversation:
            User: {query.message}
            Assistant:"""
            
            # Tokenize input
            print("Tokenizing input...")
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding='max_length'  # Ensure consistent input length
                )
                
                # Move to GPU if available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate response with timeout
                print("Generating response...")
                try:
                    # Generate with a timeout of 10 seconds using more conservative settings
                    outputs = await asyncio.wait_for(
                        asyncio.to_thread(
                            model.generate,
                            **{
                                'input_ids': inputs['input_ids'],
                                'attention_mask': inputs['attention_mask'],
                                'max_new_tokens': 100,  # Shorter responses
                                'temperature': 0.5,  # Lower temperature for more focused responses
                                'top_p': 0.85,  # More focused sampling
                                'top_k': 30,  # Limit to top 30 tokens
                                'do_sample': True,
                                'pad_token_id': tokenizer.eos_token_id,
                                'no_repeat_ngram_size': 3,  # More restrictive on repetition
                                'num_return_sequences': 1,
                                'early_stopping': True,
                                'repetition_penalty': 1.5,  # Stronger penalty for repetition
                                'length_penalty': 0.8,  # Discourage overly long responses
                                'typical_p': 0.9  # Encourage more typical/expected responses
                            }
                        ),
                        timeout=10.0  # Shorter timeout
                    )
                    
                    # Decode response
                    response = tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Extract only the assistant's response
                    if "Assistant:" in response:
                        reply = response.split("Assistant:", 1)[1].strip()
                    else:
                        reply = response.strip()
                    
                    # Clean up the response
                    reply = (
                        reply.replace('"', '')
                        .replace('\n', ' ')
                        .replace('  ', ' ')
                        .strip()
                    )
                    
                    # Remove any remaining special tokens or incomplete sentences
                    reply = ' '.join([word for word in reply.split() if not word.startswith('<') and not word.endswith('>')])
                    
                    # Ensure the response is meaningful (at least 3 words and contains letters)
                    if len(reply.split()) < 3 or not any(c.isalpha() for c in reply):
                        reply = None  # Will trigger fallback response
                    # Ensure proper punctuation
                    elif reply and reply[-1] not in {'.', '!', '?', ':', ';'}: 
                        reply = reply.rstrip(',-') + '.'
                
                except asyncio.TimeoutError:
                    error_msg = "Generation took too long. Please try again with a more specific question."
                    print(error_msg)
                    return JSONResponse(
                        status_code=408,
                        content={"error": error_msg}
                    )
                    
            except Exception as e:
                error_msg = f"Error processing your request: {str(e)}"
                print(f"Error: {error_msg}")
                print(traceback.format_exc())
                reply = None
        
        # If we still don't have a valid reply, use the default response
        if not reply:
            # Default response with suggestions
            reply = """I'm here to help with information about Map2Map's Google Maps management services. Here are some common questions I can help with:
            \n• What services do you offer?
            \n• How can you improve my Google ranking?
            \n• How do I claim my business on Google?
            \n• What makes Map2Map different?
            \n• How do I get started?
            \n• What are your business hours?
            \nFeel free to ask me anything about our services!"""
        
        print("Response generated successfully")
        print(f"Reply: {reply[:200]}...")  # Log first 200 chars
        
        return JSONResponse(content={"reply": reply})
                
    except asyncio.TimeoutError:
        error_msg = "Generation took too long. Please try again with a more specific question."
        print(error_msg)
        return JSONResponse(
            status_code=408,
            content={"error": error_msg}
        )
                
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "Sorry, I'm having trouble generating a response. Please try again."}
        )

@app.get("/openapi.json")
def get_openapi_spec():
    return FileResponse("openapi.json", media_type="application/json")

@app.get("/test")
async def test_page():
    return FileResponse("test.html")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model_name}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info",
        workers=1,  # Use only 1 worker to reduce memory usage
        limit_concurrency=10,  # Limit concurrent connections
        timeout_keep_alive=30  # Close idle connections after 30 seconds
    )
