#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
REGION="us-central1"
PROJECT="logical-light-467511-n0"
REPO="dash-repo"
IMAGE_NAME="dashboard"

NAMESPACE="aiops-dashboard"
DEPLOYMENT_NAME="k8s-dashboard"
CONTAINER_NAME="dashboard"
DEPLOYMENT_YAML="k8s/deployment.yaml"

# Sete para 1 se quiser pular o configure-docker do gcloud
: "${SKIP_DOCKER_LOGIN:=0}"
# --------------------------

artifact_host="${REGION}-docker.pkg.dev"
image_repo="us-central1-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}"

if [[ "${SKIP_DOCKER_LOGIN}" != "1" ]]; then
  echo "-> Configurando Docker para Artifact Registry (${artifact_host})..."
  gcloud auth configure-docker "${artifact_host}" -q
fi

# Lê a imagem atual do YAML
current_image="$(grep -E '^\s*image:\s*.+$' "${DEPLOYMENT_YAML}" | head -1 | awk '{print $2}')"
if [[ -z "${current_image}" ]]; then
  echo "Não encontrei a linha 'image:' em ${DEPLOYMENT_YAML}"; exit 1
fi

current_tag="${current_image##*:}"
image_path="${current_image%:*}"

# Bump de versão: suporta "1.38" (incrementa o .38), "138" (incrementa inteiro),
# caso contrário usa data+git short sha.
bump_tag() {
  local tag="$1"
  if [[ "$tag" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    local major="${BASH_REMATCH[1]}" minor="${BASH_REMATCH[2]}"
    echo "${major}.$((minor+1))"
  elif [[ "$tag" =~ ^[0-9]+$ ]]; then
    echo "$((tag+1))"
  else
    local dt sha
    dt="$(date +%Y%m%d)"
    sha="$(git rev-parse --short HEAD 2>/dev/null || echo local)"
    echo "${dt}.${sha}"
  fi
}

new_tag="$(bump_tag "${current_tag}")"
new_image="${image_repo}:${new_tag}"

echo "Imagem atual no YAML  : ${current_image}"
echo "Nova imagem (build/push): ${new_image}"

# Build + push
docker build -t "${new_image}" .
docker push "${new_image}"

# Atualiza a linha da imagem no YAML (backup .bak)
# Substitui apenas a linha que aponta para o repositório atual.
sed -i.bak "s|^\(\s*image:\s*\)${image_path}:[^[:space:]]\+|\1${new_image}|" "${DEPLOYMENT_YAML}"

# Aplica o deploy
echo "-> kubectl apply -f ${DEPLOYMENT_YAML}"
kubectl apply -f "${DEPLOYMENT_YAML}"

# Aguarda rollout
echo "-> Aguardando rollout do deployment/${DEPLOYMENT_NAME} em ${NAMESPACE}..."
kubectl -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT_NAME}" --timeout=180s

echo
echo "✅ Sucesso!"
echo "   Antiga: ${current_image}"
echo "   Nova   : ${new_image}"
echo "   YAML   : atualizado (backup em ${DEPLOYMENT_YAML}.bak)"
echo
echo "Dica: rollback rápido -> kubectl -n ${NAMESPACE} rollout undo deploy/${DEPLOYMENT_NAME}"
