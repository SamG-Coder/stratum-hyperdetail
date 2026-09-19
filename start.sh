#!/bin/sh
set -eu
cd "$(dirname "$0")"
npm run build
exec node server.mjs "$@"
