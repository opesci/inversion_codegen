pipeline {
  agent {  }
  environment {
    PATH = "/usr/local/bin:/usr/bin:/bin"
  }
  stages {
    stage 'build' {
    parallel 'gcc7-build':{
      node('xenial-gcc7'){
        checkout scm
        def customImage = docker.build("my-image:${env.BUILD_ID}")
        customImage.inside {
        sh 'ls -l'
        }
      }, 'gcc8-build':{
      node('xenial-gcc8'){
      checkout scm
        def customImage = docker.build("my-image:${env.BUILD_ID}")
        customImage.inside {
        sh 'ls -l'
        }
      }
    }
  }
}
