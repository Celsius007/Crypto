# Crypto
crypto price prediction

# Creating the back-end
#Run the following in terminal

export GCP_PROJECT_ID='wagon-bootcamp-75025' # project
export DOCKER_IMAGE_NAME="crypto"            # container
export GCR_MULTI_REGION="eu.gcr.io"          # GCP region
export GCR_REGION="europe-west1"             # GCP region 2

#Then initialize Docker and test manually
1- Launch Docker app
2- Execute docker build -t container .
3- Test locally with docker run -e PORT=8000 -p 8080:8000 container #and then go to localhost:8080

#Push to Docker if you've made modifs to the code
1- Make sure Docker is running
2- Execute docker build -t $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME .
3- Push with docker push $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME
4- Deploy to Cloud Run with gcloud run deploy --image $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME --region $GCR_REGION
