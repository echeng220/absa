from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class ReviewForm(FlaskForm):
    review = StringField('Restaurant Review', validators=[DataRequired()])

    submit = SubmitField('Submit')
