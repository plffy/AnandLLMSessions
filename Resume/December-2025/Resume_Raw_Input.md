# Anand Mohan - Resume Raw Input Data

**Purpose**: Raw information shared for resume generation. Use this as input for LLMs.

---

## Personal Information

- **Name**: Anand Mohan
- **Email**: anand.mohan164@outlook.com
- **Phone**: +91-7070265861
- **Location**: Bengaluru, India
- **LinkedIn**: https://www.linkedin.com/in/plffy/
- **Experience**: 7+ years

---

## Clear (Sep 2023 - Present)

### Role
- Working as tech lead here
- Title to use: Lead Software Engineer

### Team
- Team size started at ~5 engineers
- Now it has increased to ~20 engineers
- Mentored 10+ engineers

### E-Invoicing Platform

**Scale:**
- Processing 100M+ invoices on monthly basis
- This is average per country load seen after launching in that country and in around 6+ months of operation
- As we are launching in different countries this will multiply
- Regions: EU, MEA, SEA

**Architecture:**
- Built in microservices manner dividing things into different services
- Made architectural change that means launching multiple countries on same infra
- Country-agnostic design - single maintainable codebase to support multiple countries with distinct regulatory requirements
- Multi-tenant architecture - multiple countries on shared infrastructure while maintaining regulatory isolation
- Reduces infrastructure costs and time-to-market from months to weeks

**Country Launches:**
- Developed a plan to launch e-invoicing in a country in less than 2 weeks
- Automating different aspects with the help of LLM and automation
- Using that plan we have launched in 3 more countries: Belgium, Germany, UAE
- Planning to go live in 40+ countries within next 1 year

**PEPPOL:**
- Developed Clear Global PEPPOL-compliant services
- Implementing full Access Point functionality to send and receive invoices and related documents across multiple countries
- Ensured adherence to PEPPOL specifications
- Enabled Clear's accreditation as a trusted service provider

### GST Services Re-architecture

**What I did:**
- Re-architected the whole orchestration flow for complex GST government interaction
- Supporting flowing of ~200+ million invoices per month
- Millions while serving single request
- Did both clubbing and separation
- Carefully segregating async flows in different pods
- Setting correct retry mechanism
- Enhanced error tracking

**Results:**
- 70% higher throughput (rate of invoices processed increased)
- Reduced infrastructure costs - using less number of pods per deployment due to improved orchestration
- Eliminated OOM errors at peak load
- Eliminated slowness causing timeouts at peak load
- Reduced failure rates from 10% to <0.1%

### Metrics Framework
- Created reusable async metrics publishing framework
- For real-time dashboards
- Got adopted at Clear by other teams as well
- Organization-wide standard for service monitoring

---

## Amazon (Apr 2019 - Mar 2023)

### Role
- Software Development Engineer
- Worked as early SDE1 there - might not be relevant now

---

### Project 1: Vendor Inbound Tracking Platform

**Context:**
- Amazon has lot of vendors sending goods to the different warehouses
- There was a need to develop a service that these vendors can use to track the inbound progress of their shipment and also take some required actions on it
- Prior to this, these things used to happen manually taking lots of time in days giving bad experience to vendors
- They can't raise dispute on time leading to further operational troubles ahead

**My Role:**
- I was the lead engineer in this project responsible for designing and developing the services
- Few other new engineers worked on this project as well alongside me
- Mentored junior engineers on the team, conducting code reviews and guiding architectural decisions

**Architecture - Ingestion Service:**
- At the time actual inbounding events was getting generated for invoicing and other financial purposes
- Created an ingestion service listening to those events and storing it in our database
- SQS subscribed to SNS Event Topic
- Workers pulling in messages from SQS
- Running some data validation and data enrichment logic
- Storing it into our Database in required format based on our User Access Pattern
- Used EC2 as a worker - multiple threads was running on it to pull in the messages and process it in parallel
- Handling processing failure - for temporary exception in downstream services had implemented exponential backoff and retry strategy
- Messages having permanent exception being sent to the DLQ
- In the DLQ, we were evaluating the failure cause manually and making the required changes in the system logic

**Database Selection - DocumentDB:**
- Explored lots of database options for this
- Rejected SQL - as we were expecting changes in tables schema more frequently
- Rejected Key-Value (DynamoDB) - The access pattern was not fixed and it was supposed to evolve in future. It would have been difficult with Key-Value DB like DynamoDB with the limitation on number of LSIs and GSIs.
- Selected DocumentDB (MongoDB) - We needed flexible schema for storage, support of flexible access pattern, running at scale. Storing the data in form of document seemed like a good choice for this use-case.

**Code:**
- Ingestion Service was coded in plain Java without any framework
- Used various libraries to support the code e.g., for DI used Google Guice
- Used Chain of Responsibility Design pattern to validate and enrich the data in stepwise manner

**Web Application:**
- Frontend was developed using REACT coded in TypeScript
- Static Contents like HTML, CSS, scripts was being served to client via CloudFront (Amazon CDN)
- For Dynamic content call was being made to the backend services through API gateway

**Backend API Service:**
- Used serverless architecture for backend services
- APIs were running on AWS Lambda
- Had a fixed pool of Lambda that we kept warm to serve incoming requests with minimum delay

**Lambda Cold Start Optimization:**
- AWS Lambda has a very famous problem of Cold Start
- The time Lambda needs to be ready to serve the incoming requests includes Environment Initialisation
- Lambda was establishing DB connection after receiving the request and that was taking a lot of time
- Customised the Lambda initialisation logic to establish the DB connection during the Lambda initialisation phase only
- It helped bring down the Worst case P99 API latency from 5+ Sec to less than 1 sec
- This pattern was later reused across other projects including Amazon Go

**EU Internationalization:**
- After launching this service in India we also expanded this service to other regions of the world like EU
- Challenges: Making common data format, Scaling up the service infrastructure, Making UI Components generic so that the same code base would be shared for all the regions, Same thing for backend APIs
- Involved lots of interaction and collaboration with the EU Product and Technical Team
- Worked in away team model for implementing various features
- Solved cross-regional integration challenges
- Developed region-agnostic components
- Executed rollouts with EU teams

---

### Project 2: Amazon Go - Associate Dashboard

**Context:**
- Greenfield project
- Aimed at solving the entire journey for an associate
- From giving a portal for manual intervention
- To tracking the outputs with random assessment
- Then assigning the courses to improve the capability of the associate

**Architecture:**
- Serverless full-stack application
- S3, CloudFront, Cognito, Lambda
- Applied cold start optimizations pioneered on Vendor Portal

**CI/CD:**
- Built end-to-end CI/CD pipeline
- Infrastructure as Code using Terraform and CloudFormation
- Reduced deployment cycles from 2 days to 1 hour
- This was done for both Amazon projects

---

## Reliance Jio (Jul 2018 - Apr 2019)

- Software Engineer
- Developed software automation solutions for a web GUI-based portal
- Achieving 70% reduction in execution time

---

## Education

- B.E. Computer Science & Engineering
- Birla Institute of Technology, Mesra
- 2014 - 2018

---

## Things to NOT include

**Old Awards (not relevant at 7+ years level):**
- Smart India Hackathon Winner (2017)
- Cisco Global Cybersecurity Scholarship (2018)

**Metrics to avoid:**
- ~2000 RPM - not that good, better not mention it

**Title:**
- Don't put "Staff Software Engineer" on top - it's incorrect as not held that title
- Use "Lead Software Engineer"

---

## Technical Skills

- Languages: Java, Python, TypeScript
- Architecture: Distributed Systems, Microservices, Event-Driven, Serverless
- Cloud: AWS (Lambda, EC2, DynamoDB, SQS/SNS, S3, CloudFront), Kubernetes, Docker
- Databases: MongoDB, DocumentDB, DynamoDB, PostgreSQL, DuckDB, ClickHouse
- Infrastructure: Terraform, CloudFormation, CI/CD (Jenkins, GitHub Actions)

---

*This is raw input - use as context for resume generation*
