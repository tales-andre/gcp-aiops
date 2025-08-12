terraform {
  backend "gcs" {
    bucket  = "tfstate-logical-light-467511-n0"       # troque
    prefix  = "aiops-dashboard/prod"         # pasta/“diretório” dentro do bucket
  }
}