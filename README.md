# 🏥 Mirokaï – Hospital & Concierge Chatbot with RAG  

Mirokaï is a **Retrieval-Augmented Generation (RAG) chatbot** powered by OpenAI + LangChain.  
It demonstrates how **Large Language Models (LLMs)** can be used as helpful assistants in **hospital** and **concierge** contexts, providing accurate answers based on a structured dataset.  

This repo includes:  
- A **CLI chatbot** (`mini_chat_bot_RAG.py`)  
- A **company brochure demo** (`Enchanted_Tools_company_brochure.py`)  
- A **RAG chatbot demo for company info** (`Enchanted_Tools_company_mini_chat_bot_RAG.py`)  
- An **imaginary hospital dataset** (`california_general_hospital.json`)
- An **imaginary hotel dataset** (`Concierge_dataset.json`)  
- Supporting images and output screenshots  

⚠️ **Note:** All hospital information is **imaginary** and created only for demo/testing purposes. It is **not real medical data** and should never be used for actual healthcare guidance.  

---

## ✨ Features
- 🤖 **Two roles**: Hospital Assistant & Concierge  
- 💬 **CLI chatbot** with commands `/reset`, `/help`, `/exit`  
- 📚 **RAG integration**: retrieves answers from JSON datasets (`california_general_hospital.json`, `Concierge_dataset.json`)  
- 🧠 **Conversation memory** (remembers context across turns)  
- 🏨 **Hospital Demo**: answers questions about departments, doctors, transport, parking, etc.  
- 📄 **Company Brochure Demo**: shows brochure generation and information retrieval for "Enchanted Tools"  
- 🔍 **Vector database (ChromaDB)** for semantic search in structured data  

---

## 📂 Project Structure
```
.
├── mini_chat_bot_RAG.py                     # Main hospital/concierge chatbot with RAG
├── Enchanted_Tools_company_mini_chat_bot_RAG.py  # Company chatbot demo with RAG
├── Enchanted_Tools_company_brochure.py      # Brochure generation demo
├── california_general_hospital.json          # Imaginary hospital dataset
├── Concierge_dataset.json                    # Imaginary hotel dataset
├── healthcare-layout-of-hospital.png         # Hospital floor plan illustration
├── outputs/
│   ├── RAG_Enchanted_output_1.png            # Example RAG chatbot output
│   ├── RAG_Enchanted_output_2.png
│   ├── GPT output for Enchanted Brochure.png
│   ├── Gemini output for Enchanted Brochure.png
└── README.md                                # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mirokai-rag-chatbot.git
cd mirokai-rag-chatbot
```

### 2. Install Dependencies
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=sk-xxxxxxx
OPENAI_MODEL=gpt-4o-mini
CHAT_TEMPERATURE=0.4
```

Install requirements:
```bash
pip install -r requirements.txt
```

### 3. Run the Hospital Chatbot
```bash
python mini_chat_bot_RAG.py --role hospital --data california_general_hospital.json
```

Run in Concierge mode:
```bash
python mini_chat_bot_RAG.py --role concierge --data california_general_hospital.json
```

### 4. Run the Company RAG Demo
```bash
python Enchanted_Tools_company_mini_chat_bot_RAG.py
```

---

## 💻 Usage (Hospital Example)

```
You: hello , is there any wc in the hospital?
Mirokaï: Yes, each floor has 4 restrooms available. On the first floor, you can find them near the Emergency Department and Registration.

You: give me names of doctors in Cardiology
Mirokaï: Here are some doctors in the Cardiology Department:
1. Dr. Olivia Moore – Heart Failure Specialist
2. Dr. Mia Johnson – Electrophysiologist
3. Dr. Ethan Jackson – Interventional Cardiologist
...

You: where is the bus station?
Mirokaï: There is an onsite bus station with a shuttle to the city center every 30 minutes. The first bus departs at 7:00 AM.
```

---

## 📊 Dataset: California General Hospital  

- **Floors**: 3  
- **Beds**: 300 (100 per floor)  
- **Doctors**: 200  
- **Nurses**: 300  
- **Departments**: Emergency, Surgery, Pediatrics, Oncology, etc.  
- **Facilities**: Ambulances (100), Helipads (2), Parking (5 lots), Hotel nearby, Bus station, Taxi stand  

See [`california_general_hospital.json`](./california_general_hospital.json) for the full dataset.  

## 🏨 Dataset: California Grand Luxury Hotel

Floors: 10

Rooms: 904 (incl. luxury & presidential suites)

Staff: 850

Dining: 3 restaurants (fine dining, rooftop bar, organic café)

Facilities: 3 ballrooms, 12 meeting rooms, 4 swimming pools, tennis court, spa & wellness, cinema, live music lounge

Transport: Rooftop helipad, 3 parking lots (valet available), 24/7 taxi station, airport/downtown shuttle, limousine service

See Concierge_dataset.json
 for the full dataset.

---

## 📚 Tech Stack
- [Python 3.9+](https://www.python.org/)  
- [OpenAI API](https://platform.openai.com/) – LLM backbone  
- [LangChain](https://www.langchain.com/) – RAG orchestration  
- [ChromaDB](https://www.trychroma.com/) – Vector database  
- [Gradio](https://gradio.app/) – Web demo interface (company use case)  

---

## 🔮 Roadmap
- [ ] Add **Gradio interface** for the hospital chatbot  
- [ ] Extend to **multi-hospital datasets**  
- [ ] Add **voice input/output** for accessibility  
- [ ] Dockerize for easy deployment  

---

## ⚠️ Disclaimer
- All data in `california_general_hospital.json` is **imaginary** and **for demo only**.  
- This chatbot **does not provide medical advice**.  
- Always consult qualified professionals for healthcare.  

---

## 📜 License
MIT License – free to use, modify, and share.  
