#!/bin/bash

packages=(
    'build-essential'
    'make'
    'cmake'
    'curl'
    'git'
    'grep'
)
cmds=(
    # MODULAR LANGUAGE
    "curl https://get.modular.com | MODULAR_AUTH=mut_55c07900b0634760a280ce48dcdb3262 sh -"
    "modular install mojo"
    # git
)
function log_message() {
    local prefix="[danielnachumdev's installer]"
    local message="$1"
    echo "$prefix: $message"
}
function install_package(){
    local package="$1"
    log_message "Installing $package"
    sudo apt-get install $package
}

function install_packages(){
    for package in "${packages[@]}"; do
        install_package $package
    done
}

function execute_commands(){
    
    for cmd in "${cmds[@]}"; do
        log_message "Executing '$cmd'"
        eval $cmd
    done
    
}
function update(){
    log_message "Updating packages"
    sudo apt-get update
    log_message "Upgrading packages"
    sudo apt-get upgrade
    
}
function main(){
    install_packages
    execute_commands
    update
}

main


