from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True, device=1)
example = """EXCELENTÍSSIMO SR DR JUIZ DE DIREITO DA 6ª VARA DA FAZENDA
PÚBLICA DA COMARCA DA CAPITAL – SÃO PAULO
MANDADO DE SEGURANÇA
PROCESSO Nº 1007898-19.2022.8.26.0053

Huno Molina Rodrigues dos Santos, dentista, portador do RG nº
12.345.678-9, inscrito no CPF/MF sob nº 123.456.789-10, residente e
domiciliado na Rua dos Enforcados, nº 21, CEP 02104-010, por seu procurador,
vem propor a presente
AÇÃO DE COBRANÇA,
Em face do Município de São Paulo, pessoa jurídica de direito
público, CNPJ nº 12.345.678/0001-01, com sede no Viaduto do Chá, s/ n.
SEI 6029.2024/0016634-1
Autos 2276828-82.2024.8.26.0000 (1)
São Paulo, 30 de nov. de 24
Rui Barbosa
OAB/SP 12.345"""

ner_results = nlp(example)
for item in ner_results:
    if item["entity_group"] == "PER":
        example = example.replace(item["word"], "<NOME>")
    if item["entity_group"] == "LOC" and item["word"] != "São Paulo" :
        example = example.replace(item["word"], "<LUGAR>")
    
print (example)