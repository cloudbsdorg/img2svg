// Jenkinsfile - declarative pipeline for the img2svg project.
//
// Mirrors scripts/ci.sh so contributors can reproduce CI failures
// locally with the same commands. The Package stage is gated to
// the main branch; everything else runs on every push. The Backend
// Matrix stage runs the test suite once per supported compute
// backend (cpu, nvidia, amd, apple) so a single push exercises all
// four pyproject extras on appropriately-labeled agents.
pipeline {
    agent any

    options {
        timeout(30, 'MINUTES')
    }

    stages {
        stage('Setup') {
            steps {
                sh 'uv sync --all-extras'
            }
        }

        stage('Lint') {
            parallel {
                stage('ruff check') {
                    steps { sh 'uv run ruff check' }
                }
                stage('ruff format') {
                    steps { sh 'uv run ruff format --check' }
                }
                stage('mypy') {
                    steps { sh 'uv run mypy src/' }
                }
            }
        }

        stage('Test') {
            steps {
                sh '''uv run pytest \
                    --cov=img2svg \
                    --cov-fail-under=80 \
                    --junitxml=build/junit.xml \
                    --json-report \
                    --json-report-file=build/report.json'''
            }
        }

        stage('Test Slow') {
            when {
                branch 'main'
            }
            steps {
                sh '''uv run pytest -m slow \
                    --junitxml=build/junit-slow.xml \
                    --json-report \
                    --json-report-file=build/report-slow.json'''
            }
        }

        stage('Build Docs') {
            steps {
                sh 'uv run mkdocs build --strict'
            }
        }

        stage('Package') {
            when {
                branch 'main'
            }
            steps {
                sh 'uv build'
            }
        }

        // Per-vendor build matrix. One cell per pyproject extra
        // (cpu, nvidia, amd, apple); each cell installs the matching
        // extra, runs the fast test suite, and confirms the detected
        // backend matches the axis value. The matrix inherits
        // `agent any` from the pipeline top level; the per-vendor
        // gating (apple on macos, amd on amd-gpu, nvidia on
        // nvidia-gpu) is enforced by the verify_backend.sh step
        // rather than by per-cell agent labels, so a single generic
        // Linux agent is enough to exercise the cpu/nvidia/amd logic
        // and a macos agent picks up the apple cell.
        stage('Backend Matrix') {
            matrix {
                axes {
                    axis {
                        name 'BACKEND'
                        values 'cpu', 'nvidia', 'amd', 'apple'
                    }
                }
                stages {
                    stage('Install') {
                        steps {
                            sh 'uv pip install ".[$BACKEND]"'
                        }
                    }
                    stage('Test') {
                        steps {
                            sh 'uv run pytest -m "not slow" -q'
                        }
                    }
                    stage('Verify Backend') {
                        steps {
                            sh 'bash scripts/verify_backend.sh $BACKEND'
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            junit 'build/junit.xml'
            archiveArtifacts artifacts: 'build/,dist/,site/,htmlcov/'
        }
    }
}
