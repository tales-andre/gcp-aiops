variable "project_id"      { type = string }
variable "region" {
  type    = string
  default = "us-central1"
}


# Nomes padrões (troque se quiser)
variable "repo_name"       { 
    type = string 
    default = "dash-repo" 
    }
variable "cluster_name"    { 
    type = string 
    default = "aiops-tools" 
    }

variable "namespace"       { 
    type = string 
    default = "aiops-dashboard" 
    }