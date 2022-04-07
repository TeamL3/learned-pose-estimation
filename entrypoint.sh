#!/bin/bash
set -e

# Convenience aliases (add your own to learn from each other!)
echo "alias ll='ls -lath'" >> /home/lpe/.bashrc
echo "alias kk=clear" >> /home/lpe/.bashrc
echo "alias sourceb='source /home/lpe/.bashrc'" >> /home/lpe/.bashrc
echo "alias jl='jupyter lab'" >> /home/lpe/.bashrc
echo "alias jj='jupyter notebook'" >> /home/lpe/.bashrc
echo "alias tt='tmux new'" >> /home/lpe/.bashrc

# Setup for tmux
echo "set-option -g default-command \"exec /bin/bash\"" > ~/.tmux.conf
{
  echo "# if running bash"
  echo "  if [ -n \"$BASH_VERSION\" ]; then"
  echo "    # include .bashrc if it exists"
  echo "    if [ -f \"$HOME/.bashrc\" ]; then"
  echo "    . \"$HOME/.bashrc\""
  echo "    fi"
  echo "fi"
} >> ~/.profile

# Keybindings for jupyter notebook, feel free to add your other preferred shortcuts!
# need double quotes. so need to escape them with backslashes"
notebook_config='/home/lpe/.jupyter/nbconfig/notebook.json'
sed -i '$d' $notebook_config  # remove the last bracket in the config
mkdir -p /home/lpe/.jupyter/nbconfig/
{
  echo "  ,"
  echo "  \"keys\": {"
  echo "    \"command\": {"
  echo "      \"bind\": {"
  echo "        \"ctrl-shift-a\": \"jupyter-notebook:restart-kernel-and-run-all-cells\","
  echo "        \"ctrl-shift-q\": \"jupyter-notebook:restart-kernel-and-clear-output\""
  echo "      }"
  echo "    }"
  echo "  }"
  echo "}"
} >> $notebook_config
# add commonly used jupyter snippets via docker bind mount (see docker-comlpe.yml)
snippets='/home/lpe/.local/share/jupyter/nbextensions/snippets/'
rm $snippets/snippets.json
ln -s /snippets/snippets.json $snippets/snippets.json

exec "$@"
