import boto3

class MinioHandler:
    def __init__(self):
        # Cấu hình endpoint chỉ với cổng 9000, không thêm bất kỳ đường dẫn nào
        self.endpoint = 'http://minio:9000'
        self.access_key = "minio"  # Sử dụng access key chính xác
        self.secret_key = "tungnei12345"  # Sử dụng secret key chính xác
        self.s3_resource = self.get_s3_resource(self.endpoint, self.access_key, self.secret_key)
    
    def get_s3_resource(self, endpoint, access_key, secret_key):
        return boto3.resource(
            's3',
            endpoint_url=endpoint,  # Endpoint chỉ là URL cơ bản
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'  # Thêm region_name='us-east-1'
        )

    def upload_file_to_bucket(self, bucket_name, file_path, object_name):
        try:
            # Kiểm tra và tạo bucket nếu chưa tồn tại
            bucket = self.s3_resource.Bucket(bucket_name)
            if not bucket.creation_date:
                self.s3_resource.create_bucket(Bucket=bucket_name)
            
            # Upload file mà không cần ghi rõ URL đầy đủ
            bucket.upload_file(file_path, object_name)
            print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
        except Exception as e:
            print(f"Error uploading file to bucket: {str(e)}")
            return None
    
    