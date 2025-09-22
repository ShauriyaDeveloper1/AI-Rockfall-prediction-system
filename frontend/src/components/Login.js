import React, { useState } from 'react';
import axios from 'axios';

const Login = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    first_name: '',
    last_name: '',
    company: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [emailValid, setEmailValid] = useState(null);
  const [passwordValid, setPasswordValid] = useState(null);

  // Email validation function
  const validateEmail = (email) => {
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    if (!email) return null;
    if (!emailRegex.test(email)) return false;
    
    // Check if domain looks valid (basic check)
    const domain = email.split('@')[1];
    const validDomains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'company.com', 'university.edu'];
    const hasValidFormat = domain && domain.includes('.') && domain.length > 3;
    
    return hasValidFormat;
  };

  // Password validation function
  const validatePassword = (password) => {
    if (!password) return null;
    
    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasNumber = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    const hasMinLength = password.length >= 8;
    
    return hasUppercase && hasLowercase && hasNumber && hasSpecialChar && hasMinLength;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });

    // Real-time validation
    if (name === 'email') {
      setEmailValid(validateEmail(value));
    }
    if (name === 'password' && !isLogin) {
      setPasswordValid(validatePassword(value));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Validation before submit
    if (!isLogin) {
      if (!validateEmail(formData.email)) {
        setError('Please enter a valid email address with a proper domain');
        setLoading(false);
        return;
      }
      
      if (!validatePassword(formData.password)) {
        setError('Password must be at least 8 characters with uppercase, lowercase, number, and special character');
        setLoading(false);
        return;
      }
    }

    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const submitData = isLogin ? 
        { email: formData.email, password: formData.password } :
        { ...formData, role: 'user' }; // Default role for registration

      const response = await axios.post(`http://localhost:5000${endpoint}`, submitData);

      if (response.data.access_token) {
        // Store token and user data
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        
        // Set axios default header
        axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
        
        onLogin(response.data.user);
      } else if (!isLogin) {
        // Registration successful, switch to login
        setIsLogin(true);
        setError('Registration successful! Please login.');
        setFormData({ email: '', password: '', first_name: '', last_name: '', company: '' });
      }
    } catch (error) {
      if (error.response?.data?.details) {
        // Handle detailed validation errors from backend
        setError(error.response.data.error + ': ' + error.response.data.details.join(', '));
      } else {
        setError(error.response?.data?.error || 'An error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-6 col-lg-4">
          <div className="auth-card">
            <div className="card-header">
              <h3 className="text-center mb-0">
                {isLogin ? 'Login' : 'Register'}
              </h3>
            </div>
            <div className="card-body">
              {error && (
                <div className={`alert ${error.includes('successful') ? 'alert-success' : 'alert-danger'}`}>
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit}>
                <div className="form-group mb-3">
                  <label htmlFor="email">Email</label>
                  <input
                    type="email"
                    className={`form-control ${
                      emailValid === false ? 'is-invalid' : 
                      emailValid === true ? 'is-valid' : ''
                    }`}
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    required
                  />
                  {emailValid === false && (
                    <div className="invalid-feedback">
                      Please enter a valid email with a proper domain
                    </div>
                  )}
                  {emailValid === true && (
                    <div className="valid-feedback">
                      Valid email address
                    </div>
                  )}
                </div>

                <div className="form-group mb-3">
                  <label htmlFor="password">Password</label>
                  <input
                    type="password"
                    className={`form-control ${
                      !isLogin && passwordValid === false ? 'is-invalid' : 
                      !isLogin && passwordValid === true ? 'is-valid' : ''
                    }`}
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    required
                  />
                  {!isLogin && passwordValid === false && (
                    <div className="invalid-feedback">
                      Password must contain: 8+ characters, uppercase, lowercase, number, special character
                    </div>
                  )}
                  {!isLogin && passwordValid === true && (
                    <div className="valid-feedback">
                      Strong password!
                    </div>
                  )}
                </div>

                {!isLogin && (
                  <>
                    <div className="form-group mb-3">
                      <label htmlFor="first_name">First Name</label>
                      <input
                        type="text"
                        className="form-control"
                        id="first_name"
                        name="first_name"
                        value={formData.first_name}
                        onChange={handleInputChange}
                        required
                      />
                    </div>

                    <div className="form-group mb-3">
                      <label htmlFor="last_name">Last Name</label>
                      <input
                        type="text"
                        className="form-control"
                        id="last_name"
                        name="last_name"
                        value={formData.last_name}
                        onChange={handleInputChange}
                        required
                      />
                    </div>

                    <div className="form-group mb-3">
                      <label htmlFor="company">Company (Optional)</label>
                      <input
                        type="text"
                        className="form-control"
                        id="company"
                        name="company"
                        value={formData.company}
                        onChange={handleInputChange}
                      />
                    </div>
                  </>
                )}

                <button
                  type="submit"
                  className="btn btn-primary w-100 mb-3"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status"></span>
                      {isLogin ? 'Logging in...' : 'Registering...'}
                    </>
                  ) : (
                    isLogin ? 'Login' : 'Register'
                  )}
                </button>
              </form>

              <div className="text-center">
                <button
                  type="button"
                  className="btn btn-link"
                  onClick={() => {
                    setIsLogin(!isLogin);
                    setError('');
                    setEmailValid(null);
                    setPasswordValid(null);
                    setFormData({
                      email: '',
                      password: '',
                      first_name: '',
                      last_name: '',
                      company: ''
                    });
                  }}
                >
                  {isLogin ? "Don't have an account? Register" : "Already have an account? Login"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .auth-card {
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 20px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
          overflow: hidden;
          position: relative;
        }

        .auth-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(135deg, 
            rgba(74, 144, 226, 0.1) 0%,
            rgba(80, 200, 120, 0.1) 50%,
            rgba(255, 107, 107, 0.1) 100%);
          z-index: -1;
        }

        .card-header {
          background: rgba(255, 255, 255, 0.1);
          border-bottom: 1px solid rgba(255, 255, 255, 0.2);
          padding: 1.5rem;
        }

        .card-header h3 {
          color: #fff;
          font-weight: 600;
          text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .card-body {
          padding: 2rem;
        }

        .form-control {
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 10px;
          color: #fff;
          padding: 0.75rem 1rem;
          transition: all 0.3s ease;
        }

        .form-control:focus {
          background: rgba(255, 255, 255, 0.15);
          border-color: #4a90e2;
          box-shadow: 0 0 0 0.2rem rgba(74, 144, 226, 0.25);
          color: #fff;
        }

        .form-control.is-valid {
          border-color: #50c878;
          background: rgba(80, 200, 120, 0.1);
        }

        .form-control.is-invalid {
          border-color: #ff6b6b;
          background: rgba(255, 107, 107, 0.1);
        }

        .valid-feedback {
          display: block;
          color: #50c878;
          font-size: 0.875rem;
          margin-top: 0.25rem;
        }

        .invalid-feedback {
          display: block;
          color: #ff6b6b;
          font-size: 0.875rem;
          margin-top: 0.25rem;
        }

        .form-control::placeholder {
          color: rgba(255, 255, 255, 0.7);
        }

        label {
          color: #fff;
          font-weight: 500;
          margin-bottom: 0.5rem;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }

        .btn-primary {
          background: linear-gradient(135deg, #4a90e2 0%, #50c878 100%);
          border: none;
          border-radius: 10px;
          padding: 0.75rem 1.5rem;
          font-weight: 600;
          transition: all 0.3s ease;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .btn-primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 20px rgba(74, 144, 226, 0.3);
        }

        .btn-link {
          color: #4a90e2;
          text-decoration: none;
          font-weight: 500;
          transition: all 0.3s ease;
        }

        .btn-link:hover {
          color: #50c878;
          text-decoration: underline;
        }

        .alert {
          border: none;
          border-radius: 10px;
          backdrop-filter: blur(10px);
        }

        .alert-danger {
          background: rgba(255, 107, 107, 0.2);
          color: #ff6b6b;
          border: 1px solid rgba(255, 107, 107, 0.3);
        }

        .alert-success {
          background: rgba(80, 200, 120, 0.2);
          color: #50c878;
          border: 1px solid rgba(80, 200, 120, 0.3);
        }
      `}</style>
    </div>
  );
};

export default Login;