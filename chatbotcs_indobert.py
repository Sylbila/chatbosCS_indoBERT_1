!pip install transformers torch pandas scikit-learn

from google.colab import files
import pandas as pd

# Upload file
uploaded = files.upload()

df = pd.read_csv("faq_penerimaan.csv")
print(df.head())

model_name = "cahya/distilbert-base-indonesian"

# Load tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Pindah ke GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ Model berhasil dimuat!")

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embeddings.cpu().numpy()

# Encode semua pertanyaan FAQ
faq_questions = df['pertanyaan'].tolist()
faq_embeddings = encode_text(faq_questions)

print("Embedding FAQ selesai.")
print("Shape:", faq_embeddings.shape)  # Contoh: (10, 768)

from sklearn.metrics.pairwise import cosine_similarity

def cari_jawaban(pertanyaan_user):
    user_embedding = encode_text([pertanyaan_user])
    similarities = cosine_similarity(user_embedding, faq_embeddings)
    idx = similarities.argmax()
    max_sim = similarities[0, idx]

    if max_sim < 0.6:
        return "Maaf, saya tidak mengerti pertanyaan Anda. Silakan hubungi admin di 0812-3456-7890."

    jawaban = df.iloc[idx]['jawaban']
    return f"💬: {jawaban}"

"""pertanyaan_user = "Di mana lokasi sekolah?"
jawaban = cari_jawaban(pertanyaan_user)
print(jawaban)
"""

pertanyaan_user = "gimana cara daftar kak?"
jawaban = cari_jawaban(pertanyaan_user)
print(jawaban)

pertanyaan_user = "info beasiswa dong kak saya mau daftar dengan jalur beasiswa"
jawaban = cari_jawaban(pertanyaan_user)
print(jawaban)

pertanyaan_user = "saya mau nomor kepala sekolahnya"
jawaban = cari_jawaban(pertanyaan_user)
print(jawaban)

pertanyaan_user = "siapa nama mu?"
jawaban = cari_jawaban(pertanyaan_user)
print(jawaban)

pertanyaan_user = "gimana alur seleksinya?"
jawaban = cari_jawaban(pertanyaan_user)
print(jawaban)

pertanyaan_user = "alamat sekolah"
jawaban = cari_jawaban(pertanyaan_user)
print(jawaban)
