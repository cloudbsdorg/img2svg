// Jenkinsfile - declarative pipeline for the img2svg project.
//
// Mirrors scripts/ci.sh so contributors can reproduce CI failures
// locally with the same commands. The Package stage is gated to
// the main branch; everything else runs on every push.
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
    }

    post {
        always {
            junit 'build/junit.xml'
            archiveArtifacts artifacts: 'build/,dist/,site/,htmlcov/'
        }
    }
}
