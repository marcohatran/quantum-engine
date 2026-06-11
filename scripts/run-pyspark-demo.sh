#!/usr/bin/env bash
# Run the PySpark demo against the native engine.
# Prerequisites: native lib built (release preferred), the jar packaged
# (cd java && mvn package -DskipTests dependency:copy-dependencies
#  -DincludeScope=runtime), and pyspark==3.5.x installed (.venv works).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${JAVA_HOME:-}" && -d /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home ]]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

NATIVE_DIR="${QUANTUM_NATIVE_DIR:-}"
if [[ -z "$NATIVE_DIR" ]]; then
  if [[ -e build-release/src/libquantum_core.dylib || -e build-release/src/libquantum_core.so ]]; then
    NATIVE_DIR="$PWD/build-release/src"
  else
    NATIVE_DIR="$PWD/build-debug/src"
  fi
fi
export QUANTUM_NATIVE_DIR="$NATIVE_DIR"

DEPS=java/target/dependency
# The arrow 15 jars must precede PySpark's bundled arrow 12 (JDK 21
# compatibility) — extraClassPath entries are prepended to the driver
# classpath, so listing them here is sufficient.
# PyPI pyspark ships Scala 2.12 binaries — use the cross-built jar:
#   cd java && mvn package -Dmaven.test.skip=true \
#       -Dscala.binary.version=2.12 -Dscala.version=2.12.18 \
#   && cp target/quantum-engine-jvm-0.1.0.jar \
#         target/quantum-engine-jvm-0.1.0-scala2.12.jar
export QUANTUM_JARS="$PWD/java/target/quantum-engine-jvm-0.1.0-scala2.12.jar:$PWD/$DEPS/jna-5.14.0.jar:$PWD/$DEPS/protobuf-java-4.33.4.jar:$PWD/$DEPS/arrow-vector-15.0.2.jar:$PWD/$DEPS/arrow-memory-core-15.0.2.jar:$PWD/$DEPS/arrow-memory-netty-15.0.2.jar:$PWD/$DEPS/arrow-format-15.0.2.jar"

exec "$PYTHON" examples/quantum_pyspark_demo.py
