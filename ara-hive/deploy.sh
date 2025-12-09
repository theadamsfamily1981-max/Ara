#!/bin/bash
# =============================================================================
# Ara Hive - One-Shot Node Bootstrap
# =============================================================================
# Usage:
#   curl -sSL <repo>/deploy.sh | bash -s -- worker 192.168.1.100
#
# Or locally:
#   ./deploy.sh worker <MASTER_IP>
#   ./deploy.sh master
# =============================================================================

set -e

ROLE="${1:-worker}"
MASTER_IP="${2:-127.0.0.1}"
HIVE_DIR="${HIVE_DIR:-$HOME/ara-hive}"
DB_USER="${DB_USER:-ara}"
DB_PASS="${DB_PASS:-ara}"
DB_NAME="${DB_NAME:-ara_hive}"

echo "=== Ara Hive Bootstrap ==="
echo "Role: $ROLE"
echo "Master: $MASTER_IP"
echo "Directory: $HIVE_DIR"
echo ""

# -----------------------------------------------------------------------------
# Master setup
# -----------------------------------------------------------------------------
if [ "$ROLE" = "master" ]; then
    echo "[1/4] Setting up PostgreSQL..."

    # Check if postgres is installed
    if ! command -v psql &> /dev/null; then
        echo "Installing PostgreSQL..."
        sudo apt-get update -qq
        sudo apt-get install -y postgresql postgresql-contrib
    fi

    # Create user and database
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';" 2>/dev/null || echo "User exists"
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null || echo "Database exists"

    # Allow remote connections
    PG_HBA=$(sudo -u postgres psql -t -c "SHOW hba_file;" | xargs)
    if ! grep -q "host.*$DB_NAME.*$DB_USER.*0.0.0.0/0" "$PG_HBA" 2>/dev/null; then
        echo "host    $DB_NAME    $DB_USER    0.0.0.0/0    md5" | sudo tee -a "$PG_HBA"
        sudo systemctl reload postgresql
    fi

    # Apply schema
    echo "[2/4] Applying schema..."
    cd "$HIVE_DIR"
    PGPASSWORD="$DB_PASS" psql -U "$DB_USER" -d "$DB_NAME" -h 127.0.0.1 -f db/schema.sql

    MASTER_IP="127.0.0.1"
fi

# -----------------------------------------------------------------------------
# Common setup (both master and worker)
# -----------------------------------------------------------------------------
echo "[2/4] Setting up Python environment..."

cd "$HIVE_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

# -----------------------------------------------------------------------------
# Export DSN
# -----------------------------------------------------------------------------
export ARA_HIVE_DSN="dbname=$DB_NAME user=$DB_USER password=$DB_PASS host=$MASTER_IP"

echo "[3/4] Registering node..."
python src/register_node.py --role "$ROLE"

# -----------------------------------------------------------------------------
# Create systemd service (optional)
# -----------------------------------------------------------------------------
echo "[4/4] Creating systemd service..."

SERVICE_FILE="/etc/systemd/system/ara-bee.service"
if [ ! -f "$SERVICE_FILE" ]; then
    sudo tee "$SERVICE_FILE" > /dev/null << EOSVC
[Unit]
Description=Ara Hive Bee Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HIVE_DIR
Environment="ARA_HIVE_DSN=dbname=$DB_NAME user=$DB_USER password=$DB_PASS host=$MASTER_IP"
ExecStart=$HIVE_DIR/venv/bin/python src/bee_agent.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOSVC
    sudo systemctl daemon-reload
fi

echo ""
echo "=== Bootstrap Complete ==="
echo ""
echo "To start the bee agent manually:"
echo "  cd $HIVE_DIR && source venv/bin/activate"
echo "  export ARA_HIVE_DSN=\"$ARA_HIVE_DSN\""
echo "  python src/bee_agent.py"
echo ""
echo "Or via systemd:"
echo "  sudo systemctl start ara-bee"
echo "  sudo systemctl enable ara-bee"
echo ""

if [ "$ROLE" = "master" ]; then
    echo "To seed test tasks:"
    echo "  PGPASSWORD=$DB_PASS psql -U $DB_USER -d $DB_NAME -h 127.0.0.1 -f db/seed_dummy_tasks.sql"
    echo ""
    echo "To run evaporation (add to cron):"
    echo "  * * * * * cd $HIVE_DIR && ARA_HIVE_DSN=\"$ARA_HIVE_DSN\" venv/bin/python src/evaporate_sites.py"
fi
