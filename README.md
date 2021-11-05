# Deployment Procedure

1. Go to root folder - parent folder of app

2. Build Docker image

    '''cmd
    docker build -t absa_app .
    '''

3. Test Docker image locally

    Make sure docker image is visible:

        '''cmd
        docker images
        '''
    
    Test local Docker image:

    '''cmd
    docker run -it --rm -p 7625:80 absa_app
    '''

    Test at <http://localhost:7625/>

4. Deploy with AWS Elastic Beanstalk (requires Elastic Beanstalk CLI)

    '''cmd
    eb init
    '''

    Follow prompts, do not enable SSH or CodeCommit.

    '''cmd
    eb create
    '''

References:
https://sommershurbaji.medium.com/deploying-a-docker-container-to-aws-with-elastic-beanstalk-28adfd6e7e95