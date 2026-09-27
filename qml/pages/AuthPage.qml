import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme
import "../components" as Components

Item {
    id: root
    objectName: "authPage"

    property url heroSource: theme.backgroundUrl("auth_hero.jpg")
    readonly property bool wideLayout: width >= 1180
    readonly property bool compactLayout: height < 900 || !wideLayout
    readonly property real formCenterX: wideLayout ? metrics.authFormCenterX : width / 2
    readonly property real formLeft: Math.max(40, formCenterX - metrics.authFormWidth / 2)
    readonly property real socialGap: 30.34

    signal signUpRequested(string fullName, string email, string password)
    signal loginRequested()
    signal socialLoginRequested(string provider)

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    Rectangle {
        anchors.fill: parent
        color: theme.authBackground
    }

    Rectangle {
        id: heroPanel
        objectName: "authHeroPanel"
        visible: !root.compactLayout
        x: Math.min(metrics.authHeroX, root.width - width - metrics.authHeroY)
        y: Math.max(24, Math.min(metrics.authHeroY, (root.height - height) / 2))
        width: Math.max(420, Math.min(metrics.authHeroWidth, root.width - metrics.authHeroX - metrics.authHeroY))
        height: Math.max(420, Math.min(metrics.authHeroHeight, root.height - metrics.authHeroY * 2))
        radius: metrics.authHeroRadius
        color: theme.imagePlaceholder
        clip: true

        Image {
            anchors.fill: parent
            source: root.heroSource
            asynchronous: true
            cache: true
            fillMode: Image.PreserveAspectCrop
        }

        Components.RoundedCornerMask {
            anchors.fill: parent
            cornerRadius: metrics.authHeroRadius
            maskColor: theme.authBackground
        }
    }

    Item {
        id: referenceForm
        visible: !root.compactLayout
        anchors.fill: parent

        Image {
            x: root.formCenterX - width / 2
            y: 116
            width: metrics.authLogoWidth
            height: metrics.authLogoHeight
            source: theme.brandingUrl("logo_blue.svg")
            cache: true
            fillMode: Image.PreserveAspectFit
        }

        Row {
            x: root.formCenterX - width / 2
            y: 386.27
            spacing: root.socialGap

            Components.SocialLoginButton {
                provider: "apple"
                iconSource: theme.iconUrl("apple.svg")
                onClicked: root.socialLoginRequested(provider)
            }

            Components.SocialLoginButton {
                provider: "google"
                iconSource: theme.iconUrl("google.svg")
                onClicked: root.socialLoginRequested(provider)
            }

            Components.SocialLoginButton {
                provider: "facebook"
                iconSource: theme.iconUrl("facebook.svg")
                onClicked: root.socialLoginRequested(provider)
            }
        }

        Text {
            x: root.formCenterX - width / 2
            y: 495
            text: "or"
            color: theme.mainBlue
            font.family: typography.buttonFamily
            font.pixelSize: typography.body
        }

        Components.AuthTextField {
            id: fullNameField
            x: root.formLeft
            y: 556.41
            width: metrics.authFieldWidth
            height: metrics.authFieldHeight
            label: "Full Name"
        }

        Components.AuthTextField {
            id: emailField
            x: root.formLeft
            y: 639.27
            width: metrics.authFieldWidth
            height: metrics.authFieldHeight
            label: "Email"
        }

        Components.AuthTextField {
            id: passwordField
            x: root.formLeft
            y: 722.13
            width: metrics.authFieldWidth
            height: metrics.authFieldHeight
            label: "Password"
            passwordMode: true
        }

        Components.PrimaryButton {
            x: root.formLeft
            y: 805
            width: metrics.authPrimaryButtonWidth
            height: metrics.authPrimaryButtonHeight
            radius: metrics.authPrimaryButtonRadius
            text: "Start"
            onClicked: root.signUpRequested(fullNameField.text, emailField.text, passwordField.text)
        }

        Row {
            x: root.formCenterX - width / 2
            y: 956
            spacing: metrics.spacingTiny

            Text {
                text: "Already have an account?"
                color: theme.mainBlue
                font.family: typography.bodyFamily
                font.pixelSize: typography.body
            }

            Text {
                text: "Log In"
                color: theme.mainBlue
                font.family: typography.buttonFamily
                font.pixelSize: typography.body

                TapHandler { onTapped: root.loginRequested() }
            }
        }
    }

    ColumnLayout {
        visible: root.compactLayout
        width: Math.min(metrics.authFormWidth, root.width - 80)
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: Math.max(40, (root.width - width) / 2)
        anchors.topMargin: 48
        spacing: metrics.spacingMedium

        Image {
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredWidth: Math.min(metrics.authLogoWidth, parent.width)
            Layout.preferredHeight: metrics.authLogoHeight
            source: theme.brandingUrl("logo_blue.svg")
            cache: true
            fillMode: Image.PreserveAspectFit
        }

        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: root.socialGap

            Components.SocialLoginButton {
                provider: "apple"
                iconSource: theme.iconUrl("apple.svg")
                onClicked: root.socialLoginRequested(provider)
            }

            Components.SocialLoginButton {
                provider: "google"
                iconSource: theme.iconUrl("google.svg")
                onClicked: root.socialLoginRequested(provider)
            }

            Components.SocialLoginButton {
                provider: "facebook"
                iconSource: theme.iconUrl("facebook.svg")
                onClicked: root.socialLoginRequested(provider)
            }
        }

        Text {
            Layout.alignment: Qt.AlignHCenter
            text: "or"
            color: theme.mainBlue
            font.family: typography.buttonFamily
            font.pixelSize: typography.body
        }

        Components.AuthTextField {
            id: compactFullNameField
            Layout.fillWidth: true
            label: "Full Name"
        }

        Components.AuthTextField {
            id: compactEmailField
            Layout.fillWidth: true
            label: "Email"
        }

        Components.AuthTextField {
            id: compactPasswordField
            Layout.fillWidth: true
            label: "Password"
            passwordMode: true
        }

        Components.PrimaryButton {
            Layout.fillWidth: true
            Layout.preferredHeight: metrics.authPrimaryButtonHeight
            radius: metrics.authPrimaryButtonRadius
            text: "Start"
            onClicked: root.signUpRequested(compactFullNameField.text, compactEmailField.text, compactPasswordField.text)
        }

        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: metrics.spacingTiny

            Text {
                text: "Already have an account?"
                color: theme.mainBlue
                font.family: typography.bodyFamily
                font.pixelSize: typography.body
            }

            Text {
                text: "Log In"
                color: theme.mainBlue
                font.family: typography.buttonFamily
                font.pixelSize: typography.body

                TapHandler { onTapped: root.loginRequested() }
            }
        }
    }
}
