#!/bin/bash
set -e

# Define variables
JMETER_VERSION="5.6.3"
JMETER_TGZ="apache-jmeter-$JMETER_VERSION.tgz"
JMETER_URL="https://downloads.apache.org/jmeter/binaries/$JMETER_TGZ"
INSTALL_DIR="$HOME/apache-jmeter-$JMETER_VERSION"

# Download Apache JMeter tarball
echo "Downloading Apache JMeter $JMETER_VERSION..."
wget -c $JMETER_URL -O /tmp/$JMETER_TGZ

# Extract to home directory
echo "Extracting Apache JMeter..."
tar -xzf /tmp/$JMETER_TGZ -C $HOME

# Add JMETER bin to PATH (temp for current session)
export PATH=$INSTALL_DIR/bin:$PATH

echo "Apache JMeter $JMETER_VERSION installed at $INSTALL_DIR"
echo "Add 'export PATH=\$HOME/apache-jmeter-$JMETER_VERSION/bin:\$PATH' to your shell profile to persist the PATH."

# Verify installation
echo "Verifying JMeter version..."
jmeter --version
