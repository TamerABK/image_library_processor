import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme
import "../components" as Components

Item {
    id: root
    objectName: "authPage"

    property url heroSource: theme.backgroundUrl("auth_hero.jpg")
    // One set of controls owns the transient form state in both presentations.
    property alias fullName: fullNameField.text
    property alias email: emailField.text
    property alias password: passwordField.text
    readonly property bool wideLayout: width >= 1180
    // Keep the reference login row and a bottom margin inside the viewport.
    readonly property bool compactLayout: height < 956 + loginRow.height + 48 || !wideLayout
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

    Flickable {
        id: formViewport
        objectName: "authFormViewport"
        anchors.fill: parent
        clip: true
        contentWidth: width
        contentHeight: Math.max(height, form.y + form.height + 48)
        flickableDirection: Flickable.VerticalFlick
        boundsBehavior: Flickable.StopAtBounds
        interactive: contentHeight > height
        onContentHeightChanged: returnToBounds()
        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

        // Keyboard focus also brings fields/actions into view at short heights.
        function reveal(item) {
            const top = form.y + item.y
            const bottom = top + item.height
            if (top < contentY) {
                contentY = top
            } else if (bottom > contentY + height) {
                contentY = Math.min(contentHeight - height, bottom - height + metrics.spacingSmall)
            }
        }

        Item {
            id: form
            x: root.compactLayout
                ? (root.width - width) / 2
                : metrics.authFormCenterX - width / 2
            y: root.compactLayout ? 48 : 0
            width: Math.max(0, Math.min(metrics.authFormWidth, root.width - 80))
            height: loginRow.y + loginRow.height

            Image {
                id: formLogo
                anchors.horizontalCenter: parent.horizontalCenter
                y: root.compactLayout ? 0 : 116
                width: Math.min(metrics.authLogoWidth, parent.width)
                height: metrics.authLogoHeight
                source: theme.brandingUrl("logo_blue.svg")
                cache: true
                fillMode: Image.PreserveAspectFit
            }

            Row {
                id: socialButtons
                anchors.horizontalCenter: parent.horizontalCenter
                y: root.compactLayout ? formLogo.y + formLogo.height + metrics.spacingMedium : 386.27
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
                id: orLabel
                anchors.horizontalCenter: parent.horizontalCenter
                y: root.compactLayout ? socialButtons.y + socialButtons.height + metrics.spacingMedium : 495
                text: "or"
                color: theme.mainBlue
                font.family: typography.buttonFamily
                font.pixelSize: typography.body
            }

            Components.AuthTextField {
                id: fullNameField
                objectName: "authFullName"
                y: root.compactLayout ? orLabel.y + orLabel.height + metrics.spacingMedium : 556.41
                width: parent.width
                label: "Full Name"
                onInputActiveFocusChanged: if (inputActiveFocus) formViewport.reveal(fullNameField)
            }

            Components.AuthTextField {
                id: emailField
                objectName: "authEmail"
                y: root.compactLayout ? fullNameField.y + fullNameField.height + metrics.spacingMedium : 639.27
                width: parent.width
                label: "Email"
                onInputActiveFocusChanged: if (inputActiveFocus) formViewport.reveal(emailField)
            }

            Components.AuthTextField {
                id: passwordField
                objectName: "authPassword"
                y: root.compactLayout ? emailField.y + emailField.height + metrics.spacingMedium : 722.13
                width: parent.width
                label: "Password"
                passwordMode: true
                onInputActiveFocusChanged: if (inputActiveFocus) formViewport.reveal(passwordField)
            }

            Components.PrimaryButton {
                id: startButton
                objectName: "authStart"
                y: root.compactLayout ? passwordField.y + passwordField.height + metrics.spacingMedium : 805
                width: parent.width
                height: metrics.authPrimaryButtonHeight
                radius: metrics.authPrimaryButtonRadius
                text: "Start"
                onActiveFocusChanged: if (activeFocus) formViewport.reveal(startButton)
                onClicked: root.signUpRequested(root.fullName, root.email, root.password)
            }

            Row {
                id: loginRow
                objectName: "authLoginRow"
                anchors.horizontalCenter: parent.horizontalCenter
                y: root.compactLayout ? startButton.y + startButton.height + metrics.spacingMedium : 956
                spacing: metrics.spacingTiny

                Text {
                    text: "Already have an account?"
                    color: theme.mainBlue
                    font.family: typography.bodyFamily
                    font.pixelSize: typography.body
                }

                Text {
                    objectName: "authLoginAction"
                    text: "Log In"
                    color: theme.mainBlue
                    font.family: typography.buttonFamily
                    font.pixelSize: typography.body
                    activeFocusOnTab: true
                    Accessible.role: Accessible.Link
                    Accessible.name: text
                    Keys.onReturnPressed: root.loginRequested()
                    Keys.onSpacePressed: root.loginRequested()
                    onActiveFocusChanged: if (activeFocus) formViewport.reveal(loginRow)
                    TapHandler { onTapped: root.loginRequested() }
                }
            }
        }
    }
}
