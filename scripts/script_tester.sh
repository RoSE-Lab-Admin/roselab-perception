#!/bin/bash

# Assign the first argument to the 'name' variable,
# using "World" as the default if $1 is not provided or empty.
NAME="${1:-World}"

echo "Hello, $NAME!"
