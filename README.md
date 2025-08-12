# gcp-aiops

Este repositório contém uma aplicação Flask usada para diagnóstico e apoio operacional em clusters Kubernetes. O projeto combina coleta de logs, análise com machine learning e um chatbot capaz de sugerir soluções para erros comuns.

## Estrutura do projeto

- **app.py** – Aplicativo Flask com endpoints de diagnóstico, agregação de logs e chatbot. Inclui caches em memória para reduzir chamadas a Kubernetes e rotinas de machine learning para clusterização de mensagens.
- **dataset.json** – Base de conhecimento legada com erros e soluções detalhadas.
- **dataset_chatbot.json** – Dataset otimizado para o chatbot, com sinônimos e metadados adicionais.
- **templates/** e **static/** – Arquivos HTML e CSS usados pela interface web.
- **k8s/** – Manifests Kubernetes para deploy da aplicação.
- **build.sh** – Script que gera e publica a imagem Docker, atualizando o deployment no cluster.
- **infraestrutura/** – Módulos Terraform para provisionamento de recursos na GCP.

## Dependências

As bibliotecas principais estão listadas em `requirements.txt` e incluem Flask, Kubernetes client e pacotes de machine learning como `scikit-learn` e `faiss-cpu`.

## Funcionalidades chave

### Caching e acesso ao Kubernetes

O arquivo `app.py` utiliza um cache simples com TTL para armazenar resultados de chamadas e diminuir a carga sobre a API do cluster.

### Carregamento de datasets

Os datasets são validados e carregados na inicialização da aplicação, permitindo a geração de respostas padronizadas para erros detectados nos logs.

### Endpoints do chatbot

O chatbot possui endpoints para reconstruir o índice, checar saúde e responder perguntas baseadas nos dados catalogados.

### Automação de build e deploy

O script `build.sh` realiza o build da imagem Docker, publica no Artifact Registry e atualiza o deployment Kubernetes, aguardando o rollout.

## Execução local

1. Instale as dependências: `pip install -r requirements.txt`.
2. Inicie o aplicativo Flask: `python app.py`.
3. Acesse via navegador em `http://localhost:8080`.

## Testes

O projeto ainda não possui testes automatizados. Rode `pytest` para confirmar.

