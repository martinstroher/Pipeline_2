import os
import sys
import datetime
import re
import pymupdf4llm
import pathlib
import pymupdf
import unidecode
import multiprocessing
import json
import ocrmypdf
from multiprocessing import Pool
from src.utils import log
from tqdm import tqdm

#=============================================================================
def strLimpa(str_text):
    str_limpo = re.sub(r'[\!"#•$ªº§%&\*+<=>?@^_`\[\]|~=´]', ' ', str_text)
    str_limpo = re.sub(r'/',' ', str_limpo)
    str_limpo = re.sub(r'\s+',' ',str_limpo)
    str_limpo = str_limpo.rstrip("-")
    return(str_limpo.strip())
#fim def
#=============================================================================
def limpas_erros_ocr(str_text):
    lista_suja = ["-","=","|","(",")","[","]","@","#","%","¨","&","*","ª","º","~","^","`","´","?","!",":",";",".",",","§","•"]
    str_limpo = str_text

    for caracter in lista_suja:
        str_limpo = re.sub(r"caracter+",'',str_limpo)
        str_limpo = re.sub(r"\'+",'',str_limpo)
    #fim for
    return(str_limpo)
#fim def
#=============================================================================
def calcula_total_paginas(doc_pdf):
    paginas = [pags for pags in range(0,len(doc_pdf))]
    total_paginas = len(paginas)
    return(paginas,total_paginas)
#fim def
#=============================================================================
def calcula_total_imagens(doc_pdf,pags):
    lista_imagens = 0
    total_imagens = 0
    for numero_pagina in pags:
        pagina = doc_pdf[numero_pagina] # leitura da pagina
        lista_imagens = pagina.get_images()
        total_imagens += len(lista_imagens)
    #fim for
    return(lista_imagens,total_imagens)
#fim def
#=============================================================================
def e_numero_regex(str):
    digito = re.sub(r'[\.,-]','',str)
    return(digito.isdigit())
#fim def
#=============================================================================
def extrai_texto(doc_pdf,pags):
    texto_doc = []

    for numero_pagina in pags:
        pagina = doc_pdf[numero_pagina] # leitura da pagina
        estrutura_blocos = pagina.get_text("blocks")

        for bloco in estrutura_blocos:
            tipo_bloco = int(bloco[6])

            if (tipo_bloco == 0):
                bloco_quebrado = bloco[4].split("\n")
                blocos_limpos = []

                for string_bloco in bloco_quebrado:
                    string_bloco = string_bloco.strip()
                    if (string_bloco != ""):
                        if not(e_numero_regex(string_bloco)):
                            bloco_limpo = strLimpa(string_bloco)
                            blocos_limpos.append(bloco_limpo)
                        #fim if
                    #fim if
                #fim for
            #fim if

            if (blocos_limpos):
                texto_pagina = ' '.join(blocos_limpos)
                texto_doc.append(texto_pagina)
            #fim if
        #fim for
    #fim for

    if (len(texto_doc) != 0):
        return('\n'.join(texto_doc))
    else: return(False)
#fim def
#=============================================================================
def salva_texto_final(local,nome,txt):
    arquivo_texto_final = open(local+nome+".txt.ia", "w")
    arquivo_texto_final.write(txt)
    arquivo_texto_final.close()
#fim def
#=============================================================================
def extrai_texto_ocr(doc_pdf,pags):
    texto_doc_ocr = []

    for numero_pagina in pags:
        pagina = doc_pdf[numero_pagina] # leitura da pagina
        texto_ocr = pagina.get_textpage_ocr()
        estrutura_ocr = pagina.get_text("dict",textpage=texto_ocr)

        for bloco in estrutura_ocr["blocks"]:
            blocos_limpos = []

            for linha in bloco['lines']:
                try:
                    spans_txt = linha['spans']
                    for span in spans_txt:
                        string_bloco = span['text'].strip()

                        if (string_bloco != ""):
                            if not(e_numero_regex(string_bloco)):
                                bloco_limpo = strLimpa(string_bloco)
                                bloco_limpo = limpas_erros_ocr(bloco_limpo)
                                blocos_limpos.append(bloco_limpo)
                            #fim if
                        #fim if
                    #fim for
                except:
                    log.error("File access error during OCR extraction")
                #fim try
            #fim for

            if (blocos_limpos):
                texto_pagina = ' '.join(blocos_limpos)
                texto_doc_ocr.append(texto_pagina)
            #fim if
        #fim for
    #fim for

    if (len(texto_doc_ocr) != 0):
        return('\n'.join(texto_doc_ocr))
    else: return(False)
#fim def
#=============================================================================
def processa_pdfs(pasta,lista_pdfs,total_pdfs):
    numero_cpus = multiprocessing.cpu_count()
    #print("CPUs:",numero_cpus)
    mp = Pool(numero_cpus)
    contador = 1
    pars = ()
    work = ()

    for arquivo_pdf in tqdm(lista_pdfs):
        v_partes_nome = arquivo_pdf.split(".pdf")
        if not(os.path.exists(pasta+v_partes_nome[0]+".txt.ia")):
            pars = ([pasta,arquivo_pdf,contador,total_pdfs],)
            work += pars
            contador += 1
        #else: print("\n\tArquivo:",pasta+arquivo_pdf,"processado anteriormente!")
        #fim if
    #fim for

    if (len(pars) > 0):
        resultado_doc = mp.map(processa_doc_pdf,work)
        return(resultado_doc)
    else: return False
#fim def
#=============================================================================
def processa_doc_pdf(pars):
    #*** VERIFICANDO ESTRUTURA DO DOCUMENTO ***
    #print("Processando arquivo: ",pars[2],"de",pars[3],"("+pars[1]+")")

    try:
        documento_pdf = pymupdf.open(pars[0]+pars[1])
    except:
        vetor_resultado["nc"].append(pars[1])
        log.error(f"PDF conversion failed: {pars[1]}")
    else:
        paginas,total_paginas = calcula_total_paginas(documento_pdf)
        lista_imagens,total_imagens = calcula_total_imagens(documento_pdf,paginas)
        vetor_resultado = {"c":[],"ocrs":[],"nc":[]}

        #Verificando se as paginas sao imagens de OCR
        if (total_paginas == total_imagens):
            vetor_resultado["ocrs"].append(pars[1])
            texto_extraido = ""
            texto_extraido = extrai_texto_ocr(documento_pdf,paginas)
        else:
            texto_extraido = ""
            texto_extraido = extrai_texto(documento_pdf,paginas)
        #fim if

        if (texto_extraido):
            v_partes_nome = pars[1].split(".pdf")
            salva_texto_final(pars[0],v_partes_nome[0],texto_extraido)
            vetor_resultado["c"].append(pars[1])
            salva_md(pars[0]+v_partes_nome[0])
        else: vetor_resultado["nc"].append(pars[1])
    #fim try

    #print("\tTempo parcial: ",datetime.datetime.now())
    return(vetor_resultado)
#fim def
#=============================================================================
def salva_md(arquivo):
    md_text = pymupdf4llm.to_markdown(arquivo+".pdf")
    pathlib.Path(arquivo+".md").write_bytes(md_text.encode())
#fim def
#=============================================================================
def processa_ocr_pdf(pars):
    pasta = pars[0]
    arquivo_destino = pars[1]+".ocr"
    #print("\n=> Processando OCR:",pars[1])

    if (os.path.exists(pars[0]+arquivo_destino)):
        #print("\n\t=> Arquivo processado anteriormente!")
        return(arquivo_destino)
    else:
        try:
            ocrmypdf.ocr(pars[0]+pars[1],pars[0]+arquivo_destino,l="por+eng",output_type="pdf",redo_ocr=True,tesseract_timeout=30,skip_big=1,force_ocr=True)
        except:
            #print("\tErro de OCR:",pars[1])
            return(pars[1])
        else:
            if (os.path.exists(pars[0]+arquivo_destino)):
                return(arquivo_destino)
            else: return(pars[1])
        #fim try
    #fim else
#fim def
#=============================================================================
def prepara_ocrs(pasta,lista_pdfs,total_pdfs):
    numero_cpus = multiprocessing.cpu_count()
    #print("CPUs:",numero_cpus)
    mp = Pool(numero_cpus)
    contador = 1
    pars = ()
    work = ()

    for arquivo_pdf in tqdm(lista_pdfs):
        pars = ([pasta,arquivo_pdf,contador,total_pdfs],)
        work += pars
        contador += 1
    #fim for

    resultado_ocr = mp.map(processa_ocr_pdf,work)
    return(resultado_ocr)
#fim def
#=============================================================================
def extract_text_for_rag_test(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    lista_arquivos_pdf = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    for arquivo_pdf in tqdm(lista_arquivos_pdf):
        input_path = os.path.join(input_folder, arquivo_pdf)
        output_path = os.path.join(output_folder, arquivo_pdf.replace('.pdf', '.txt'))
        
        try:
            documento_pdf = pymupdf.open(input_path)
            paginas, _ = calcula_total_paginas(documento_pdf)
            texto_extraido = extrai_texto(documento_pdf, paginas)
            
            if texto_extraido:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(texto_extraido)
                log.detail(f"Extracted: {arquivo_pdf}")
            else:
                log.warn(f"Text extraction failed: {arquivo_pdf}")
        except Exception as e:
            log.error(f"Processing {arquivo_pdf}: {e}")
#fim def
#=============================================================================
#=============================================================================
def process_folder(pasta_textos):
    """
    Main function to process all PDFs in a folder using the Robust pipeline.
    Arguments:
        pasta_textos: Path to the folder containing PDFs (must end with /)
    """
    if not pasta_textos.endswith("/"):
        pasta_textos += "/"

    if not os.path.exists(pasta_textos):
        log.error(f"Folder {pasta_textos} does not exist.")
        return

    log.info("Scanning folder for PDF files...")
    lista_arquivos_pdf = []
    
    for nome_arquivo in sorted(os.listdir(pasta_textos)):
        if nome_arquivo.endswith('.pdf'):
            lista_arquivos_pdf.append(nome_arquivo)

    total_arquivos_pdf = len(lista_arquivos_pdf)
    log.info(f"Found {total_arquivos_pdf} PDF files.")
    
    if total_arquivos_pdf == 0:
        return

    lista_arquivos_imagens_ocr = []
    lista_arquivos_erros = []
    lista_arquivos_convertidos = []
    lista_master = []

    numero_cpus = multiprocessing.cpu_count()
    log.detail(f"Using {numero_cpus} CPU cores")
    indice_inicial = 0
    while indice_inicial <= total_arquivos_pdf:
        indice_final = indice_inicial + numero_cpus
        subconjunto = lista_arquivos_pdf[indice_inicial:indice_final]
        lista_master.append(subconjunto)
        indice_inicial += numero_cpus

    contador_lista = 1
    for lista_processar in tqdm(lista_master):
        lista_arquivos_ocrs = prepara_ocrs(pasta_textos,lista_processar,total_arquivos_pdf)

        if (lista_arquivos_ocrs):
            resultados = processa_pdfs(pasta_textos,lista_arquivos_ocrs,total_arquivos_pdf)

            if (resultados):
                for resultado in resultados:
                    for c in resultado["c"]:
                        lista_arquivos_convertidos.append(c)
                    for nc in resultado["nc"]:
                        lista_arquivos_erros.append(nc)
                    for ocr in resultado["ocrs"]:
                        lista_arquivos_imagens_ocr.append(ocr)

            # Logging results...
            with open(pasta_textos+"arquivos_convertidos_"+str(contador_lista)+".json", "w") as final:
                json.dump(lista_arquivos_convertidos, final)
            with open(pasta_textos+"arquivos_nao_convertidos_"+str(contador_lista)+".json", "w") as final:
                json.dump(lista_arquivos_erros, final)
            with open(pasta_textos+"arquivos_convertidos_ocr_"+str(contador_lista)+".json", "w") as final:
                json.dump(lista_arquivos_imagens_ocr, final)

            final = datetime.datetime.now()
            str_parcial = "Convertidos:"+str(len(lista_arquivos_convertidos))+"\nNao Convertidos:"+str(len(lista_arquivos_erros))+"\nConvertidos OCR:"+str(len(lista_arquivos_imagens_ocr))+"\nTempo Parcial:"+str(final)+"\n"
            arquivo_parcial = open(pasta_textos+"resultado_parcial_"+str(contador_lista)+".dat", "a")
            arquivo_parcial.write(str_parcial)
            arquivo_parcial.close()

            contador_lista += 1
        else: log.warn("No OCR files to convert.")

    str_final = "Convertidos:"+str(len(lista_arquivos_convertidos))+"\nNao Convertidos:"+str(len(lista_arquivos_erros))+"\nConvertidos OCR:"+str(len(lista_arquivos_imagens_ocr))+"\n"
    arquivo_final = open(pasta_textos+"resultado_final.dat", "a")
    arquivo_final.write(str_final)
    arquivo_final.close()

#=============================================================================

if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'rag_test':
        input_folder = sys.argv[2]
        output_folder = sys.argv[3]
        extract_text_for_rag_test(input_folder, output_folder)
    
    elif len(sys.argv) >= 2:
        pasta_textos = sys.argv[1]
        if len(sys.argv) == 3:
             # Logic for single file not fully refactored, assuming folder mode for integration
             pass 
        else:
             process_folder(pasta_textos)
    else:
        log.error("Invalid syntax!")
        log.info("Usage: python pdf_processor.py <pdf-folder>/")
