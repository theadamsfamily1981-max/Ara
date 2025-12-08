# =============================================================================
# Ara VPN POP Infrastructure
# =============================================================================
# Dual-protocol VPN Point of Presence
# WireGuard (fast path) + OpenVPN (compatibility)
#
# Usage:
#   terraform init
#   terraform plan -var="pop_name=ara-pop-us-east"
#   terraform apply
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "region" {
  description = "AWS region for POP deployment"
  type        = string
  default     = "us-east-1"
}

variable "pop_name" {
  description = "Name of the VPN POP"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "allowed_admin_cidrs" {
  description = "CIDRs allowed SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production!
}

variable "ssh_public_key" {
  description = "SSH public key for admin access"
  type        = string
  default     = ""
}

variable "wireguard_port" {
  description = "WireGuard UDP port"
  type        = number
  default     = 51820
}

variable "openvpn_port" {
  description = "OpenVPN UDP port"
  type        = number
  default     = 1194
}

variable "enable_openvpn" {
  description = "Enable OpenVPN compatibility mode"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

# =============================================================================
# Provider
# =============================================================================

provider "aws" {
  region = var.region
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# =============================================================================
# Security Group
# =============================================================================

resource "aws_security_group" "vpn_pop" {
  name        = "${var.pop_name}-sg"
  description = "Ara VPN POP security group"
  vpc_id      = data.aws_vpc.default.id

  # WireGuard (always enabled)
  ingress {
    description = "WireGuard"
    from_port   = var.wireguard_port
    to_port     = var.wireguard_port
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # OpenVPN (optional)
  dynamic "ingress" {
    for_each = var.enable_openvpn ? [1] : []
    content {
      description = "OpenVPN"
      from_port   = var.openvpn_port
      to_port     = var.openvpn_port
      protocol    = "udp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  # SSH admin access
  ingress {
    description = "SSH Admin"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_admin_cidrs
  }

  # Prometheus metrics (internal only)
  ingress {
    description = "Prometheus Node Exporter"
    from_port   = 9100
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }

  # WireGuard exporter
  ingress {
    description = "WireGuard Exporter"
    from_port   = 9586
    to_port     = 9586
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.pop_name}-sg"
    Role = "ara-vpn-pop"
  })
}

# =============================================================================
# SSH Key Pair
# =============================================================================

resource "aws_key_pair" "vpn_pop" {
  count      = var.ssh_public_key != "" ? 1 : 0
  key_name   = "${var.pop_name}-key"
  public_key = var.ssh_public_key

  tags = merge(var.tags, {
    Name = "${var.pop_name}-key"
  })
}

# =============================================================================
# EC2 Instance
# =============================================================================

resource "aws_instance" "vpn_pop" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.vpn_pop.id]
  key_name               = var.ssh_public_key != "" ? aws_key_pair.vpn_pop[0].key_name : null

  # Enable source/dest check disable for VPN routing
  source_dest_check = false

  root_block_device {
    volume_size           = 20
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  # Bootstrap script - installs Python for Ansible
  user_data = <<-EOF
    #!/bin/bash
    set -e
    apt-get update
    apt-get install -y python3 python3-apt software-properties-common

    # Enable IP forwarding
    echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf
    echo 'net.ipv6.conf.all.forwarding=1' >> /etc/sysctl.conf
    sysctl -p

    # Tag completion
    echo "Bootstrap complete" > /var/log/ara-bootstrap.log
  EOF

  tags = merge(var.tags, {
    Name        = var.pop_name
    Role        = "ara-vpn-pop"
    Environment = "production"
    ManagedBy   = "terraform"
  })

  lifecycle {
    ignore_changes = [ami]  # Don't replace on AMI updates
  }
}

# =============================================================================
# Elastic IP
# =============================================================================

resource "aws_eip" "vpn_pop" {
  instance = aws_instance.vpn_pop.id
  domain   = "vpc"

  tags = merge(var.tags, {
    Name = "${var.pop_name}-eip"
  })
}

# =============================================================================
# Outputs
# =============================================================================

output "pop_name" {
  description = "VPN POP name"
  value       = var.pop_name
}

output "public_ip" {
  description = "Public IP address of VPN POP"
  value       = aws_eip.vpn_pop.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.vpn_pop.id
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.vpn_pop.id
}

output "wireguard_endpoint" {
  description = "WireGuard endpoint"
  value       = "${aws_eip.vpn_pop.public_ip}:${var.wireguard_port}"
}

output "openvpn_endpoint" {
  description = "OpenVPN endpoint (if enabled)"
  value       = var.enable_openvpn ? "${aws_eip.vpn_pop.public_ip}:${var.openvpn_port}" : "disabled"
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh ubuntu@${aws_eip.vpn_pop.public_ip}"
}

output "ansible_inventory_entry" {
  description = "Entry for Ansible inventory"
  value       = "${var.pop_name} ansible_host=${aws_eip.vpn_pop.public_ip} ansible_user=ubuntu"
}
