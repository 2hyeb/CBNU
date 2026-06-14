
import warnings
warnings.filterwarnings('ignore')

checks = []

try:
    from sentence_transformers import SentenceTransformer
    checks.append('sentence_transformers: OK')
except Exception as e:
    checks.append(f'sentence_transformers: {e}')

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    checks.append('lc_community HF embed: OK')
except Exception as e:
    checks.append(f'lc_community HF embed: {type(e).__name__}')

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    checks.append('langchain_openai: OK')
except Exception as e:
    checks.append(f'langchain_openai: {e}')

try:
    import ragas
    checks.append(f'ragas version: {ragas.__version__}')
except Exception as e:
    checks.append(f'ragas: {e}')

for c in checks:
    print(c)
