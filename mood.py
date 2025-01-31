import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import logging
from nltk.corpus import wordnet
from random import shuffle, randint, random, choice
from sklearn.decomposition import PCA
import nltk

# Certifique-se de baixar os dados do NLTK necessários
nltk.download("wordnet")
nltk.download("omw-1.4")

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Diretório para salvar embeddings e metadados
output_dir = "embeddings"
os.makedirs(output_dir, exist_ok=True)

# Arquivo de metadados
metadata_file = os.path.join(output_dir, "metadata.csv")
if not os.path.exists(metadata_file):
    with open(metadata_file, "w") as f:
        f.write("filename,song,artist,genre\n")  # Cabeçalho do CSV

# Carregar dados
lyrics_data = pd.read_csv(
    "/home/jmayos/songtest/music_lyrics.csv"
)  # CSV contendo letras

# Carregar o modelo LLaMA
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="cpu")

# Configurar o token de preenchimento, se necessário
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Função para limpar o texto da letra
def clean_text(text):
    """Limpar o texto da letra, removendo caracteres especiais e convertendo para minúsculas."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# Função para substituir palavras por sinônimos
def synonym_replacement(text):
    words = text.split()
    augmented_text = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms and random() > 0.7:  # Substituir com 30% de probabilidade
            synonym = choice(synonyms).lemmas()[0].name()
            augmented_text.append(synonym.replace("_", " "))
        else:
            augmented_text.append(word)
    return " ".join(augmented_text)


# Função para reorganizar linhas da letra
def shuffle_lyrics(text):
    lines = text.split("\n")
    shuffle(lines)
    return "\n".join(lines)


# Função para truncar ou expandir texto
def truncate_or_expand(text):
    words = text.split()
    if random() > 0.5:
        return " ".join(words[: len(words) // 2])  # Truncar
    return " ".join(words + words[: len(words) // 2])  # Expandir


# Função para adicionar ruído textual
def add_noise(text):
    words = text.split()
    for _ in range(randint(1, 3)):  # Adicionar até 3 palavras aleatórias
        index = randint(0, len(words) - 1)
        words.insert(index, choice(["na", "la", "yeah", "oh"]))
    return " ".join(words)


# Função principal de data augmentation
def augment_text(text):
    augmented_versions = [
        synonym_replacement(text),
        shuffle_lyrics(text),
        truncate_or_expand(text),
        add_noise(text),
    ]
    return [clean_text(aug) for aug in augmented_versions if aug.strip()]


def reduce_dimension(embedding, target_dim=128):
    """
    Reduz as dimensões do embedding selecionando diretamente as primeiras dimensões.
    """
    if embedding.shape[1] > target_dim:
        logging.info(f"Reducing embedding dimensions from {embedding.shape[1]} to {target_dim}")
        return embedding[:, :target_dim]
    else:
        logging.warning(f"Embedding já possui dimensão menor ou igual ao alvo ({target_dim}): {embedding.shape}")
        return embedding



# Alternativa para redução de dimensão com média global
def reduce_dimension_with_mean(embedding):
    return embedding.mean(axis=1)


def get_lyrics_embeddings(lyrics_text, target_dim=128):
    """
    Gerar embedding da letra usando representações ocultas do modelo LLaMA.
    Aplica redução de dimensionalidade para adequar o tamanho do embedding.
    """
    inputs = tokenizer(lyrics_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Última camada oculta
        embeddings = hidden_states.mean(dim=1).cpu().numpy()  # Média ao longo do tempo

    # Reduzir apenas o último eixo
    embeddings = reduce_dimension(embeddings, target_dim=target_dim)
    return embeddings



# Atualizar o processo de geração de embeddings
def process_song_with_augmentation(index, row):
    try:
        song_title = row["Song"]
        artist_name = row["Artist"]
        lyrics = row["Lyrics"]
        genre = row["Genre"]

        if pd.isnull(lyrics) or lyrics.lower() == "letra não encontrada.":
            logging.warning(f"Skipping song '{song_title}' by '{artist_name}' due to missing lyrics.")
            return 0

        cleaned_lyrics = clean_text(lyrics)

        logging.info(f"Processing song '{song_title}' by '{artist_name}' with augmentations.")
        logging.info(f"Genre: {genre}")

        # Gera embedding da letra original e variações aumentadas
        all_lyrics_versions = [cleaned_lyrics] + augment_text(cleaned_lyrics)
        embeddings_count = 0

        for idx, version in enumerate(all_lyrics_versions):
            # Gerar embedding para cada versão aumentada com redução de dimensionalidade
            lyrics_embeddings = get_lyrics_embeddings(version, target_dim=128)

            # Salvar embedding
            embedding_filename = f"song_{index}_aug_{idx}_embedding.npy"
            np.save(os.path.join(output_dir, embedding_filename), lyrics_embeddings)

            # Salvar metadados
            with open(metadata_file, "a") as f:
                f.write(f"{embedding_filename},{song_title}_aug_{idx},{artist_name},{genre}\n")

            embeddings_count += 1

        logging.info(f"Generated {embeddings_count} embeddings for '{song_title}'.")
        return embeddings_count

    except Exception as e:
        logging.error(f"Error processing song '{row.get('Song', 'Unknown')}': {e}")
        return 0


# Processar músicas com data augmentation
def process_lyrics_with_augmentation():
    total_embeddings = 0
    for index, row in lyrics_data.iterrows():
        total_embeddings += process_song_with_augmentation(index, row)
    logging.info(f"Processing complete. Generated {total_embeddings} embeddings.")
    return total_embeddings


# Executar o processamento com data augmentation
augmented_embeddings_count = process_lyrics_with_augmentation()

logging.info(f"Total embeddings (with augmentations): {augmented_embeddings_count}")
