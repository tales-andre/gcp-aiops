

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------- Artifact Registry (Docker) ----------
resource "google_artifact_registry_repository" "repo" {
  provider      = google
  location      = var.region
  repository_id = var.repo_name
  format        = "DOCKER"
  description   = "Imagens do dashboard"
}


#########################
# Vars e provider
#########################
variable "project" {
  description = "ID do projeto GCP"
  type        = string
}




# IMPORTANTE: nas VPCs auto-mode (inclusive a 'default'), o nome da sub-rede costuma ser a PRÓPRIA região
# ex.: us-central1. Se no seu projeto for diferente, ajuste aqui.
variable "default_subnet_name" {
  description = "Nome da sub-rede existente na VPC default para a região escolhida"
  type        = string
  default     = "us-central1"
}



#########################
# Usa a VPC 'default' existente
#########################
data "google_compute_network" "default" {
  name = "default"
}

# Usa a sub-rede default existente na mesma região do cluster
# (você confirmou que o nome é "default" em us-central1)
data "google_compute_subnetwork" "default" {
  name   = "default"
  region = var.region
}


resource "google_container_cluster" "default" {
  name             = var.cluster_name
  location         = var.region
  enable_autopilot = true

  # Usa VPC/sub-rede existentes
  network    = data.google_compute_network.default.self_link
  subnetwork = data.google_compute_subnetwork.default.self_link

  # Deixe o Autopilot gerenciar as faixas secundárias automaticamente.
  # (Não defina ip_allocation_policy aqui.)

  # Autopilot exige logging ligado e logging de system components
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS"]
  }

  # Autopilot exige monitoring com Managed Service for Prometheus habilitado
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  deletion_protection = false
}