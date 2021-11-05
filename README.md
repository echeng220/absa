# Deployment Procedure

1. Go to root folder - parent folder of app

2. Build Docker image

    '''cmd
    docker build -t absa_app .
    '''

3. Test Docker image locally

    '''cmd
    docker run -it --rm -p 7625:80 absa_app
    '''

    Test at <http://localhost:7625/>

4.