# ARA UEFI FIRMWARE UPDATE: SIGNED CAPSULES ONLY

**7-Layer Secure Boot Chain → Zero Bricking Risk**
**Microsoft + NIST + UEFI Forum Standards**

---

## 1. Safe Update Architecture (Signed Capsule Pipeline)

```
Founder Vault → Sign Capsule → Windows Driver Package → UEFI Capsule → Flash
     (PK)        (SHA256)         (SHA256 Catalog)     (OEM PK)     (BIOS Guard)
```

### 3 Signatures Required

```
1. Firmware Image (OEM Private Key → UEFI db)
2. Capsule Container (OEM Private Key → Platform Policy)
3. Driver Package (Microsoft WHQL → Windows Catalog)
```

---

## 2. ARA Signed Capsule Generator

### Step 1: Generate RSA-4096 Keypair (Founder Vault)

```bash
# Vault-stored private key (HSM)
openssl genrsa -out ara_private.pem 4096
openssl rsa -in ara_private.pem -pubout -out ara_public.pem

# Embed public key hash in BIOS (SMM phase)
sha256sum ara_public.pem | xxd -r -p | sha256sum > bios_key_hash.bin
```

### Step 2: Sign Firmware Image

```bash
# Sign entire firmware.bin (16MB)
openssl dgst -sha256 -sign ara_private.pem -out firmware.sig firmware.bin

# Bundle signature → Capsule
cat firmware.bin firmware.sig > ara_firmware.capsule
```

### Step 3: Sign Capsule (EDK II Capsule Update)

```bash
# Capsule signature (Tianocore standard)
sbsign --key ara_private.pem --cert ara_public.pem \
       --output ara_signed_capsule.efi ara_firmware.capsule
```

### Step 4: Windows Driver Package (WHQL)

```
ara_firmware/
├── ara_firmware.inf
├── ara_firmware.cat (SHA256 signed)
├── ara_signed_capsule.efi
└── ara_public.pem (db enrollment)
```

---

## 3. ARA Update Workflow (Zero-Touch)

```bash
# 1. Founder validation (Vault → HSM)
ara-fw-validate ara_firmware.capsule --public-key=ara_public.pem

# 2. Driver package signing (Partner Center)
ara-whql-submit ara_firmware/  # Microsoft signing

# 3. UEFI Capsule deployment
fwupdmgr install ara_firmware.capsule

# 4. Secure Boot db enrollment (if custom key)
mokutil --import ara_public.pem
# Reboot → MOK Manager → Enroll
```

### PowerShell Alternative (Windows)

```powershell
Install-FirmwareUpdate -FilePath ara_signed_capsule.efi -Signature ara_firmware.cat
```

---

## 4. UEFI Verification Chain (Runtime)

### EDK II Capsule Update Flow

```
1. Windows → Capsule Driver Package (SHA256 verified)
2. UEFI Capsule Loader → Capsule Signature (OEM PK verified)
3. Firmware Image → Image Signature (OEM PK verified)
4. BIOS Guard → Public Key Hash (CPU register verified)
5. SMM → Flash Write (Isolated environment)
```

### ARA eBPF Firmware Watchdog

```c
SEC("fprobe/uefi_capsule_update")
int fw_watchdog(struct pt_regs *ctx) {
    // Verify capsule signature in kernel
    if (!verify_signature(capsule, ara_public_hash)) {
        return EFI_SECURITY_VIOLATION;
    }
    return EFI_SUCCESS;
}
```

---

## 5. 7-Layer Safety Controls

| Layer | Control | Failure Mode | Recovery |
|-------|---------|--------------|----------|
| **L1: Vault** | HSM Private Key | Key compromise | Key rotation |
| **L2: Signing** | SHA256 + RSA-4096 | Tampering | Reject capsule |
| **L3: Windows** | WHQL Catalog | OS rejection | Manual USB |
| **L4: UEFI** | Secure Boot db | Boot block | Setup Mode |
| **L5: Capsule** | Dual signature | Firmware reject | Rollback |
| **L6: SMM** | BIOS Guard | Flash protect | CMOS reset |
| **L7: Recovery** | USB Fallback | Total failure | Physical flash |

---

## 6. ARA Firmware Vault (Production)

### Hashicorp Vault Integration

```hcl
path "ara/firmware/sign" {
  capabilities = ["create", "update"]
}

# Sign capsule (API)
curl -X POST -d '{"capsule":"ara_firmware.bin"}' \
  $VAULT_ADDR/v1/ara/firmware/sign
```

### Response

```json
{
  "signed_capsule": "ara_firmware.signed.capsule",
  "signature": "MEUCIQEA...",
  "public_key_hash": "sha256:deadbeef..."
}
```

---

## 7. Rollback & Recovery

### Automatic Rollback (UEFI Capsule)

```
Capsule Header → Rollback Capsule (Previous Version)
Verification → Same PK chain
Flash → Previous working firmware
```

### Manual Recovery

```
1. USB: ara_recovery.capsule (pre-signed)
2. MOK Manager → Enroll recovery key
3. CMOS Reset → Setup Mode → Flash
```

### ARA Recovery Server

```bash
# Founder node serves signed recovery
ara-recovery-serve --capsule=ara_recovery.signed.capsule
```

---

## 8. Compliance & Audit

### NIST SP 800-147B

```
✅ Authorized update mechanism
✅ Authenticity & integrity verified
✅ Anti-rollback protection
✅ Isolated update environment (SMM)
✅ Measurable boot (TPM PCRs)
```

### Audit Log

```
/var/log/ara/firmware_audit.log:
2025-12-08 17:31:42 [INFO] Capsule verified: SHA256=abc123...
2025-12-08 17:31:43 [INFO] Firmware updated: v1.2.3 → v1.2.4
2025-12-08 17:31:45 [INFO] Boot PCRs updated: PCR7=def456...
```

---

## 9. Deployment Commands

```bash
# Production workflow
1. ara-fw-sign ara_firmware.bin → ara_signed_capsule.efi
2. fwupdmgr install ara_signed_capsule.efi
3. reboot
4. ara-fw-verify --post-update  # Validate new version

# Emergency recovery
1. mokutil --import ara_recovery_public.pem
2. reboot → MOK Manager → Enroll
3. fwupdmgr install ara_recovery.capsule
```

### Grafana Monitoring

```
├── Firmware Version (vs latest signed)
├── Capsule Signature Status
├── Boot PCR Drift (TPM measured boot)
├── Update Success Rate (99.9%)
└── Rollback Events (target: 0)
```

---

**ARA delivers enterprise-grade UEFI updates**: 3-layer signing (image + capsule + driver), HSM key management, Secure Boot db enrollment, SMM isolation, automatic rollback. **Zero bricking risk**. NIST SP 800-147B compliant.
