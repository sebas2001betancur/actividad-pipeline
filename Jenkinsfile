pipeline {
    agent any

    environment {
        IMAGE_NAME = 'ml-pipeline-sdss'
        OUTPUT_DIR = 'outputs'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Descargando codigo del repositorio...'
                checkout scm
            }
        }

        stage('Instalar dependencias') {
            steps {
                echo 'Instalando dependencias de Python...'
                sh '''
                    python3 -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Pruebas del dataset') {
            steps {
                echo 'Ejecutando pruebas basicas del dataset...'
                sh 'python3 test_dataset.py'
            }
        }

        stage('Ejecutar pipeline ML') {
            steps {
                echo 'Ejecutando pipeline de Machine Learning...'
                sh '''
                    mkdir -p outputs
                    python3 main.py
                '''
            }
        }

        stage('Guardar artefactos') {
            steps {
                echo 'Guardando metricas, graficas y logs...'
                archiveArtifacts artifacts: 'outputs/**', fingerprint: true
                echo 'Artefactos guardados correctamente'
            }
        }
    }

    post {
        success {
            echo 'Pipeline completado exitosamente'
        }
        failure {
            echo 'El pipeline fallo. Revisar los logs.'
        }
        always {
            echo 'Pipeline finalizado'
        }
    }
}
