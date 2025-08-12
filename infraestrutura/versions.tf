terraform {
  required_version = ">= 1.4"
  required_providers {
    google     = { source = "hashicorp/google",     version = ">= 5.12" }
    kubernetes = { source = "hashicorp/kubernetes", version = ">= 2.30" }
  }
}
