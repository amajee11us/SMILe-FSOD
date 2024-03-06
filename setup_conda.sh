#!/bin/bash

# Install Anaconda
echo "Setting up Anaconda environment..."

# Specify the Anaconda version
ANACONDA_VERSION="Anaconda3-2021.11-Linux-x86_64.sh"

# Download the Anaconda installer
wget https://repo.anaconda.com/archive/$ANACONDA_VERSION

# Run the Anaconda installer
chmod +x $ANACONDA_VERSION
./$ANACONDA_VERSION -b -p $HOME/anaconda

# Remove the installer to save space
rm $ANACONDA_VERSION

# Add Anaconda to PATH in .bashrc
echo 'export PATH="$HOME/anaconda/bin:$PATH"' >> $HOME/.bashrc

# Source .bashrc to update the PATH
source $HOME/.bashrc

# Initialize Conda for shell interaction
conda init

echo "Anaconda setup is complete."

# Verify Anaconda installation
conda --version